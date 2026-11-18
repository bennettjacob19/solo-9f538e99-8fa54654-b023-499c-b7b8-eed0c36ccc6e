"""SQLite 持久化层（aiosqlite，单连接 + WAL + 异步互斥锁）。

两个设计要点共同保证构建登记的原子性与查询一致性：

1. 登记事务互斥：``create_build_with_steps`` 在 ``self._lock`` 保护下用
   ``BEGIN IMMEDIATE ... COMMIT`` 把 build 与全部 steps 一次性落库。
   持锁期间，任何查询/状态更新都无法插入事务中间 —— 未落定的构建对外
   不可见；登记失败或进程崩溃，事务整体回滚，不留缺步的可见构建。

2. 复合操作串行化：每个查询的 execute→fetch→close 在锁内原子完成。
   否则并发协程会在某查询的 fetch 与 close 之间插入新语句，而旧语句的
   读事务尚未结束，新语句会"加入"该读事务、读到旧快照（stale read）。
   游标即取即闭，让读事务在锁释放前确定结束，不依赖 GC 时机。
"""
from __future__ import annotations

import asyncio
import contextlib
import shutil
from typing import Any, Iterable, Optional

import aiosqlite

from . import config

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


# 登记用步骤记录：(step_uid, command, depends_on_json, timeout_seconds)
StepSpec = tuple[str, str, str, float]


class Database:
    def __init__(self) -> None:
        self._conn: Optional[aiosqlite.Connection] = None
        # 复合操作互斥锁：登记事务与每条查询都完整持锁，防止交错
        self._lock = asyncio.Lock()

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
            with contextlib.suppress(Exception):
                await self._conn.close()
            self._conn = None

    # ---- 基础操作（加锁版本供外部调用；下划线版本假定调用方已持锁） -------

    async def _exec(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        assert self._conn is not None
        cur = await self._conn.execute(sql, params)
        try:
            return cur.lastrowid
        finally:
            await cur.close()

    async def _one(self, sql: str, params: tuple[Any, ...] = ()) -> Optional[aiosqlite.Row]:
        assert self._conn is not None
        cur = await self._conn.execute(sql, params)
        try:
            return await cur.fetchone()
        finally:
            # 必须关闭游标：未走到 SQLITE_DONE 的 SELECT 会让读事务挂起，
            # 之后的语句会加入该读事务而读到旧快照
            await cur.close()

    async def _all(self, sql: str, params: tuple[Any, ...] = ()) -> list[aiosqlite.Row]:
        assert self._conn is not None
        cur = await self._conn.execute(sql, params)
        try:
            return list(await cur.fetchall())
        finally:
            await cur.close()

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        async with self._lock:
            return await self._exec(sql, params)

    async def one(self, sql: str, params: tuple[Any, ...] = ()) -> Optional[aiosqlite.Row]:
        async with self._lock:
            return await self._one(sql, params)

    async def all(self, sql: str, params: tuple[Any, ...] = ()) -> list[aiosqlite.Row]:
        async with self._lock:
            return await self._all(sql, params)

    # ---- 业务封装 -------------------------------------------------------

    async def create_build_with_steps(
        self, name: str, now: str, steps: Iterable[StepSpec]
    ) -> int:
        """原子登记：build 行与全部 step 行在同一事务落库。

        整个事务持锁进行：落定前查询接口看不到该构建（查询在锁外等待，
        只能读到落定前或落定后的完整状态），调度器也不会被启动；
        任一步失败（含日志目录/文件创建失败）即整体回滚，并清理已创建的
        日志目录，不留缺步构建或文件系统残留。进程在事务进行中崩溃时，
        SQLite 会在下次打开时自动回滚未提交事务，效果相同。
        """
        specs = list(steps)
        build_id: Optional[int] = None
        async with self._lock:
            await self._exec("BEGIN IMMEDIATE")
            try:
                build_id = await self._exec(
                    "INSERT INTO builds (name, status, created_at) VALUES (?, 'pending', ?)",
                    (name, now),
                )
                log_dir = config.LOG_ROOT / str(build_id)
                log_dir.mkdir(parents=True, exist_ok=True)
                for step_uid, command, depends_on, timeout in specs:
                    log_path = log_dir / f"{step_uid}.log"
                    log_path.touch()
                    await self._exec(
                        """
                        INSERT INTO steps
                            (build_id, step_uid, command, depends_on, timeout_seconds,
                             status, log_path)
                        VALUES (?, ?, ?, ?, ?, 'pending', ?)
                        """,
                        (build_id, step_uid, command, depends_on, timeout, str(log_path)),
                    )
                await self._exec("COMMIT")
            except BaseException:
                with contextlib.suppress(Exception):
                    await self._exec("ROLLBACK")
                if build_id is not None:
                    shutil.rmtree(config.LOG_ROOT / str(build_id), ignore_errors=True)
                raise
        return build_id

    async def get_build(self, build_id: int) -> Optional[aiosqlite.Row]:
        return await self.one("SELECT * FROM builds WHERE id=?", (build_id,))

    async def list_builds(self, limit: int = 100) -> list[aiosqlite.Row]:
        return await self.all(
            "SELECT * FROM builds ORDER BY id DESC LIMIT ?", (limit,)
        )

    async def list_all_build_ids(self) -> set[int]:
        rows = await self.all("SELECT id FROM builds")
        return {row["id"] for row in rows}

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
        from .models import STEP_TERMINAL

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
        return await self.all(
            "SELECT * FROM builds WHERE status NOT IN ('succeeded','failed','cancelled','interrupted')"
        )

    async def list_steps_for_builds(self, build_ids: tuple[int, ...]) -> list[aiosqlite.Row]:
        if not build_ids:
            return []
        placeholders = ",".join("?" * len(build_ids))
        return await self.all(
            f"SELECT * FROM steps WHERE build_id IN ({placeholders}) ORDER BY id",
            build_ids,
        )
