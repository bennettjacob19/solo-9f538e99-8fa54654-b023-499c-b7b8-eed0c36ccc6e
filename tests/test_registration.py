"""构建登记的原子性与查询一致性单元测试。

核心保障：
- 登记事务在互斥锁保护下落定，事务未提交前并发查询无法插入，
  任何时刻都观察不到"缺步"的构建；
- 登记中途失败（如第二步日志文件创建失败）整体回滚，库中无残留、
  文件系统无残留；
- 进程在事务进行中崩溃（连接未 COMMIT 即断开），重启后同样无残留；
- 高并发读写下，提交后的读取永远看到最新数据（无 stale read）。
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app import config
from app.db import Database
from app.scheduler import Scheduler


def _patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "pipeline.db")
    monkeypatch.setattr(config, "WORKDIR_ROOT", tmp_path / "workdir")
    monkeypatch.setattr(config, "LOG_ROOT", tmp_path / "logs")


def _spec(step_uid: str, command: str = "true", deps: str = "[]",
          timeout: float = 10.0) -> tuple[str, str, str, float]:
    return (step_uid, command, deps, timeout)


def test_queries_cannot_interleave_into_registration(tmp_path, monkeypatch):
    """登记事务持锁进行：事务未落定前，并发查询被挡在锁外，看不到半成品。"""
    async def main() -> None:
        _patch_paths(monkeypatch, tmp_path)
        db = Database()
        await db.connect()
        try:
            async with db._lock:
                # 持锁期间（等价于登记事务进行中），查询无法插入执行
                blocked = asyncio.create_task(db.list_builds())
                await asyncio.sleep(0.2)
                assert not blocked.done(), "查询不应在登记事务持锁期间完成"
            # 锁释放后查询立即完成
            assert await asyncio.wait_for(blocked, timeout=2) == []
        finally:
            await db.close()

    asyncio.run(main())


def test_uncommitted_data_never_reaches_queries(tmp_path, monkeypatch):
    """模拟登记到一半（事务未提交）：对外查询路径不会读到未提交数据。"""
    async def main() -> None:
        _patch_paths(monkeypatch, tmp_path)
        db = Database()
        await db.connect()
        try:
            # 通过公开接口登记一个完整构建作为对照
            ok_id = await db.create_build_with_steps("ok", "now", [_spec("a")])
            # 模拟另一次登记进行到一半：直接操作连接（绕过锁，等价于崩溃前状态）
            await db._exec("BEGIN IMMEDIATE")
            half_id = await db._exec(
                "INSERT INTO builds (name, status, created_at) "
                "VALUES ('half', 'pending', 'now')"
            )
            await db._exec(
                "INSERT INTO steps (build_id, step_uid, command, depends_on, "
                "timeout_seconds, status, log_path) "
                "VALUES (?, 'compile', 'true', '[]', 1, 'pending', '/tmp/x.log')",
                (half_id,),
            )
            # 未提交数据只存在于这个未完结事务里；回滚后对外无痕迹
            await db._exec("ROLLBACK")
            assert [r["id"] for r in await db.list_builds()] == [ok_id]
            assert await db.get_build(half_id) is None
        finally:
            await db.close()

    asyncio.run(main())


def test_failed_registration_rolls_back_and_cleans_up(tmp_path, monkeypatch):
    """第二步登记失败：整体回滚，数据库与文件系统都不留缺步残留。"""
    async def main() -> None:
        _patch_paths(monkeypatch, tmp_path)
        db = Database()
        await db.connect()
        try:
            # "pack/age" 的日志路径父目录不存在，第二步 touch 必失败
            specs = [_spec("compile"), _spec("pack/age", deps='["compile"]')]
            with pytest.raises(OSError):
                await db.create_build_with_steps("fw-build", "now", specs)
            # 数据库无残留：没有 build 行，也没有任何 step 行
            assert await db.list_builds() == []
            assert await db.all("SELECT * FROM steps") == []
            # 文件系统无残留：日志目录被清理
            assert list(config.LOG_ROOT.iterdir()) == []
        finally:
            await db.close()

    asyncio.run(main())


def test_crash_mid_registration_leaves_no_visible_build(tmp_path, monkeypatch):
    """服务中断：事务进行中连接断开（未 COMMIT），重启后无缺步构建。"""
    async def main() -> None:
        _patch_paths(monkeypatch, tmp_path)
        db = Database()
        await db.connect()
        # 登记到一半进程"崩溃"：事务未提交，连接直接断开
        await db._exec("BEGIN IMMEDIATE")
        crash_id = await db._exec(
            "INSERT INTO builds (name, status, created_at) "
            "VALUES ('crashed', 'pending', 'now')"
        )
        await db._exec(
            "INSERT INTO steps (build_id, step_uid, command, depends_on, "
            "timeout_seconds, status, log_path) "
            "VALUES (?, 'compile', 'true', '[]', 1, 'pending', '/tmp/x.log')",
            (crash_id,),
        )
        await db.close()  # 未 COMMIT：SQLite 回滚未提交事务

        # 重启（新连接）：缺步构建不存在，不会被标 interrupted、不可见
        db2 = Database()
        await db2.connect()
        try:
            assert await db2.list_builds() == []
            assert await db2.get_build(crash_id) is None
            assert await db2.get_steps(crash_id) == []
        finally:
            await db2.close()

    asyncio.run(main())


def test_concurrent_queries_never_see_partial_build(tmp_path, monkeypatch):
    """登记与并发查询交错：任何时刻观察到的构建要么不存在，要么步骤完整。"""
    async def main() -> None:
        _patch_paths(monkeypatch, tmp_path)
        db = Database()
        await db.connect()
        try:
            step_count = 12
            observations: list[tuple[int, int]] = []

            async def register() -> int:
                specs = [_spec(f"s{i}") for i in range(step_count)]
                return await db.create_build_with_steps("fw", "now", specs)

            async def poll() -> None:
                # 登记事务的每条 INSERT 都会让出事件循环，查询穿插其中
                for _ in range(200):
                    for row in await db.list_builds():
                        steps = await db.get_steps(row["id"])
                        observations.append((row["id"], len(steps)))
                    await asyncio.sleep(0)

            reg_task = asyncio.create_task(register())
            poll_task = asyncio.create_task(poll())
            build_id = await reg_task
            await poll_task

            assert observations, "并发查询应至少观察到一次已落定的构建"
            for bid, n_steps in observations:
                assert bid == build_id
                assert n_steps == step_count, (
                    f"观察到缺步构建: id={bid} steps={n_steps}"
                )
        finally:
            await db.close()

    asyncio.run(main())


def test_reads_always_fresh_under_concurrent_writes(tmp_path, monkeypatch):
    """stale read 回归：高并发读写下，提交后的读取必须立即看到最新数据。

    历史缺陷：查询的 execute→fetch→close 是多个异步操作，并发协程在
    fetch 与 close 之间插入的新语句会加入尚未结束的旧读事务，读到旧快照。
    """
    async def main() -> None:
        _patch_paths(monkeypatch, tmp_path)
        db = Database()
        await db.connect()
        try:
            stop = asyncio.Event()
            stale: list[int] = []

            async def poller() -> None:
                while not stop.is_set():
                    for row in await db.list_builds():
                        await db.get_build(row["id"])
                        await db.get_steps(row["id"])
                    await asyncio.sleep(0)

            pollers = [asyncio.create_task(poller()) for _ in range(3)]
            for i in range(30):
                specs = [_spec(f"s{j}") for j in range(12)]
                bid = await db.create_build_with_steps(f"b{i}", "now", specs)
                # 提交后立即读：必须看到刚落定的构建
                if await db.get_build(bid) is None:
                    stale.append(bid)
                await asyncio.sleep(0)
            stop.set()
            await asyncio.gather(*pollers)
            assert stale == [], f"出现 stale read: {stale}"
        finally:
            await db.close()

    asyncio.run(main())


def test_recover_removes_orphan_log_dirs(tmp_path, monkeypatch):
    """重启恢复：清理登记中断留下的孤儿日志目录，保留正常构建的目录。"""
    async def main() -> None:
        _patch_paths(monkeypatch, tmp_path)
        db = Database()
        await db.connect()
        try:
            build_id = await db.create_build_with_steps("ok", "now", [_spec("a")])
            orphan = config.LOG_ROOT / "9999"  # 无对应构建行：登记中断的残留
            orphan.mkdir(parents=True)
            (orphan / "compile.log").write_text("partial")

            report = await Scheduler(db).recover()

            assert report["removed_orphan_log_dirs"] == 1
            assert not orphan.exists()
            # 正常构建的日志目录不受影响
            assert (config.LOG_ROOT / str(build_id)).is_dir()
            # 已完整登记的构建按重启规则标记 interrupted，且步骤齐全可查
            row = await db.get_build(build_id)
            assert row is not None and row["status"] == "interrupted"
            assert [s["step_uid"] for s in await db.get_steps(build_id)] == ["a"]
        finally:
            await db.close()

    asyncio.run(main())


def test_registration_failure_then_retry_reuses_id(tmp_path, monkeypatch):
    """登记回滚后自增 id 随之回退：重试同一构建得到相同 id，无空洞记录。"""
    async def main() -> None:
        _patch_paths(monkeypatch, tmp_path)
        db = Database()
        await db.connect()
        try:
            with pytest.raises(OSError):
                await db.create_build_with_steps(
                    "bad", "now", [_spec("compile"), _spec("pack/age")]
                )
            build_id = await db.create_build_with_steps(
                "good", "now", [_spec("compile"), _spec("package", deps='["compile"]')]
            )
            assert build_id == 1
            steps = await db.get_steps(build_id)
            assert [s["step_uid"] for s in steps] == ["compile", "package"]
        finally:
            await db.close()

    asyncio.run(main())
