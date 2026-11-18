"""SQLite 持久化层（aiosqlite，单连接 + WAL）。

全库只有一个连接：所有读写在连接线程上串行，杜绝跨连接的锁/快照冲突。
构建登记在该连接上用显式单事务完成，并由一把 asyncio 写锁把
"登记事务"与其他写操作互斥 —— 否则其他协程的写入会混入登记事务，
随回滚一起丢失。事务提交前，同连接的查询能看到未提交行，因此查询接口
必须过滤 registering 状态（见 get_build / list_builds）。
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, Callable, Optional

import aiosqlite

from . import config
from .models import REGISTERING_STATUS, STEP_TERMINAL, StepIn

_SCHEMA = """
CREATE TABLE IF NOT EXISTS builds (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL,
    status           TEXT NOT NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    detail           TEXT,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    build_id        INTEGER NOT NULL,
    step_uid        TEXT NOT NULL,
    command         TEXT NOT NULL,
    depends_on      TEXT NOT NULL,
    timeout_seconds REAL NOT NULL,
    status          TEXT NOT NULL,
    exit_code       INTEGER,
    detail          TEXT,
    log_path        TEXT,
    pid             INTEGER,
    pgid            INTEGER,
    proc_starttime  REAL,
    started_at      TEXT,
    finished_at     TEXT,
    UNIQUE(build_id, step_uid),
    FOREIGN KEY(build_id) REFERENCES builds(id)
);
"""


class Database:
    def __init__(self) -> None:
        self._conn: Optional[aiosqlite.Connection] = None
        # 写锁：登记事务（BEGIN...COMMIT）期间，其他协程的写必须等待，
        # 否则其语句会混入登记事务，随回滚丢失。3.10+ 的 Lock 惰性绑定事件循环。
        self._write_lock = asyncio.Lock()

    async def connect(self) -> None:
        config.ensure_dirs()
        self._conn = await aiosqlite.connect(
            config.DB_PATH, timeout=30, isolation_level=None
        )
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA busy_timeout=30000")
        await self._conn.executescript(_SCHEMA)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        """不加锁的执行原语：仅登记事务内部（已持锁）使用。"""
        assert self._conn is not None
        cur = await self._conn.execute(sql, params)
        try:
            return cur.lastrowid
        finally:
            await cur.close()

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        """所有写操作的入口：写锁序列化，避免混入登记事务。"""
        async with self._write_lock:
            return await self._execute(sql, params)

    async def one(self, sql: str, params: tuple[Any, ...] = ()) -> Optional[aiosqlite.Row]:
        assert self._conn is not None
        cur = await self._conn.execute(sql, params)
        try:
            return await cur.fetchone()
        finally:
            await cur.close()

    async def all(self, sql: str, params: tuple[Any, ...] = ()) -> list[aiosqlite.Row]:
        assert self._conn is not None
        cur = await self._conn.execute(sql, params)
        try:
            return list(await cur.fetchall())
        finally:
            await cur.close()

    # ---- 业务封装 -------------------------------------------------------

    async def create_build_with_steps(
        self,
        name: str,
        now: str,
        steps: list[StepIn],
        path_factory: Callable[[int, str], str],
    ) -> int:
        """单事务登记整张 DAG：构建行 + 全部步骤行一次落定。

        事务序列：BEGIN -> INSERT builds(registering) -> INSERT 全部 steps
        -> UPDATE builds 置 pending -> COMMIT。任一步失败即整体回滚，
        服务中断时未提交事务由 SQLite 自动回滚，因此已提交数据里
        不存在 registering 状态、也不存在缺步构建。

        整个事务持有写锁：其他协程的写操作（如调度器落状态）在锁外等待，
        不会混入本事务；同连接的并发读能看到未提交行，由查询侧的
        registering 过滤保证其不可见。
        """
        async with self._write_lock:
            await self._execute("BEGIN IMMEDIATE")
            try:
                build_id = await self._execute(
                    "INSERT INTO builds (name, status, created_at) VALUES (?, ?, ?)",
                    (name, REGISTERING_STATUS, now),
                )
                for step in steps:
                    await self._execute(
                        """
                        INSERT INTO steps
                            (build_id, step_uid, command, depends_on, timeout_seconds,
                             status, log_path)
                        VALUES (?, ?, ?, ?, ?, 'pending', ?)
                        """,
                        (
                            build_id,
                            step.id,
                            step.command,
                            json.dumps(step.depends_on),
                            step.timeout_seconds,
                            path_factory(build_id, step.id),
                        ),
                    )
                # 同一事务内翻转可见性：COMMIT 后构建即完整可见
                await self._execute(
                    "UPDATE builds SET status='pending' WHERE id=?", (build_id,)
                )
                await self._execute("COMMIT")
            except BaseException:
                with contextlib.suppress(Exception):
                    await self._execute("ROLLBACK")
                raise
            return build_id

    async def purge_registering_builds(self) -> int:
        """删除登记未完成的构建残骸（防御性清理，正常为 0）。

        事务保证 registering 状态不会被提交；此处兜底应对任何意外写入，
        保证缺步构建永远不可见、也不会被恢复逻辑当作中断构建。
        """
        rows = await self.all(
            "SELECT id FROM builds WHERE status=?", (REGISTERING_STATUS,)
        )
        if not rows:
            return 0
        ids = tuple(row["id"] for row in rows)
        placeholders = ",".join("?" * len(ids))
        await self.execute(
            f"DELETE FROM steps WHERE build_id IN ({placeholders})", ids
        )
        await self.execute(
            f"DELETE FROM builds WHERE id IN ({placeholders})", ids
        )
        return len(ids)

    async def get_build(self, build_id: int) -> Optional[aiosqlite.Row]:
        # 登记中的构建（registering）对外不可见；双保险：登记事务本就不会被读到
        return await self.one(
            "SELECT * FROM builds WHERE id=? AND status != ?",
            (build_id, REGISTERING_STATUS),
        )

    async def list_builds(self, limit: int = 100) -> list[aiosqlite.Row]:
        return await self.all(
            "SELECT * FROM builds WHERE status != ? ORDER BY id DESC LIMIT ?",
            (REGISTERING_STATUS, limit),
        )

    async def get_steps(self, build_id: int) -> list[aiosqlite.Row]:
        return await self.all(
            "SELECT * FROM steps WHERE build_id=? ORDER BY id", (build_id,)
        )

    async def get_step(self, build_id: int, step_uid: str) -> Optional[aiosqlite.Row]:
        return await self.one(
            "SELECT * FROM steps WHERE build_id=? AND step_uid=?",
            (build_id, step_uid),
        )

    async def set_build_status(
        self,
        build_id: int,
        status: str,
        now: str,
        detail: Optional[str] = None,
        started: bool = False,
        finished: bool = False,
    ) -> None:
        sets = ["status=?"]
        params: list[Any] = [status]
        if detail is not None:
            sets.append("detail=?")
            params.append(detail)
        if started:
            sets.append("started_at=COALESCE(started_at, ?)")
            params.append(now)
        if finished:
            sets.append("finished_at=?")
            params.append(now)
        params.append(build_id)
        await self.execute(f"UPDATE builds SET {', '.join(sets)} WHERE id=?", tuple(params))

    async def request_cancel(self, build_id: int) -> None:
        await self.execute(
            "UPDATE builds SET cancel_requested=1 WHERE id=?", (build_id,)
        )

    async def set_step_process(
        self, step_pk: int, pid: int, pgid: int, starttime: float, now: str
    ) -> None:
        await self.execute(
            "UPDATE steps SET pid=?, pgid=?, proc_starttime=?, status='running', "
            "started_at=COALESCE(started_at, ?) WHERE id=?",
            (pid, pgid, starttime, now, step_pk),
        )

    async def set_step_status(
        self,
        step_pk: int,
        status: str,
        now: str,
        exit_code: Optional[int] = None,
        detail: Optional[str] = None,
    ) -> None:
        sets = ["status=?"]
        params: list[Any] = [status]
        # finished_at 只在进入终态时落笔；cleaning_up 等中间态不写
        if status in STEP_TERMINAL:
            sets.append("finished_at=?")
            params.append(now)
        if exit_code is not None:
            sets.append("exit_code=?")
            params.append(exit_code)
        if detail is not None:
            sets.append("detail=?")
            params.append(detail)
        params.append(step_pk)
        await self.execute(f"UPDATE steps SET {', '.join(sets)} WHERE id=?", tuple(params))

    async def list_nonterminal_builds(self) -> list[aiosqlite.Row]:
        # registering 不属于"非终态旧构建"：它是未完成的登记，由 purge 处理
        return await self.all(
            "SELECT * FROM builds WHERE status NOT IN "
            "('succeeded','failed','cancelled','interrupted') AND status != ?",
            (REGISTERING_STATUS,),
        )

    async def list_steps_for_builds(self, build_ids: tuple[int, ...]) -> list[aiosqlite.Row]:
        if not build_ids:
            return []
        placeholders = ",".join("?" * len(build_ids))
        return await self.all(
            f"SELECT * FROM steps WHERE build_id IN ({placeholders}) ORDER BY id",
            build_ids,
        )
