"""构建登记原子性的组件级测试（进程内，不起 HTTP 服务）。

覆盖：
- 正常登记：事务提交后构建完整可见、步骤齐全、可正常调度跑完；
- 登记进行中：并发查询（列表/详情）不得看到该构建 —— 对应
  "提交尚未结束，列表也能看到它" 的原始缺陷；
- 登记中途失败（模拟"编译→打包"第二步登记失败）：API 报错且
  数据库零残留，重启恢复不会把它当作中断构建；
- 防御性清理：registering 残骸在启动恢复时被删除且永不出现；
- 并发登记：多个提交各自完整，互不串扰。
"""
from __future__ import annotations

import asyncio
import sqlite3

import pytest

from app import config
from app.db import Database
from app.models import BuildCreate, StepIn
from app.scheduler import Scheduler


def _payload(name: str = "fw", steps: int = 2) -> BuildCreate:
    return BuildCreate(
        name=name,
        steps=[
            StepIn(id=f"s{i}", command="true", depends_on=[], timeout_seconds=5)
            for i in range(steps)
        ],
    )


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """隔离数据目录下已连接的 Database。"""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "pipeline.db")
    monkeypatch.setattr(config, "WORKDIR_ROOT", tmp_path / "workdir")
    monkeypatch.setattr(config, "LOG_ROOT", tmp_path / "logs")
    database = Database()
    asyncio.run(database.connect())
    yield database
    asyncio.run(database.close())


def test_submit_registers_complete_build(db):
    async def main():
        sched = Scheduler(db)
        bid = await sched.submit(_payload(steps=3))
        rows = await db.list_builds()
        assert [r["id"] for r in rows] == [bid]
        assert rows[0]["status"] in ("pending", "running", "succeeded")
        steps = await db.get_steps(bid)
        assert len(steps) == 3
        assert {s["step_uid"] for s in steps} == {"s0", "s1", "s2"}
        # 登记完成后调度正常接管：等任务跑完，构建应成功
        await asyncio.gather(*sched._build_tasks.values())
        row = await db.get_build(bid)
        assert row["status"] == "succeeded"

    asyncio.run(main())


def test_build_invisible_until_registration_commits(db, monkeypatch):
    """登记事务进行中，并发查询不得暴露该构建。

    单连接下同事务的未提交行对并发读可见，本测试直接验证查询侧的
    registering 过滤：事务已插入构建行但尚未提交时，列表/详情都查不到。
    """
    entered = asyncio.Event()
    release = asyncio.Event()
    original = Database._execute

    async def gating(self, sql, params=()):
        result = await original(self, sql, params)
        if "INSERT INTO builds" in sql:
            entered.set()
            await release.wait()  # 模拟"提交尚未结束"的窗口
        return result

    monkeypatch.setattr(Database, "_execute", gating)

    async def main():
        sched = Scheduler(db)
        submit_task = asyncio.create_task(sched.submit(_payload(steps=2)))
        await asyncio.wait_for(entered.wait(), timeout=5)

        # 登记进行中（构建行已插入、事务未提交）：列表与详情都必须看不到
        assert await db.list_builds() == []
        assert await db.get_build(1) is None

        release.set()
        bid = await asyncio.wait_for(submit_task, timeout=10)
        rows = await db.list_builds()
        assert len(rows) == 1 and rows[0]["id"] == bid
        assert len(await db.get_steps(bid)) == 2  # 可见即完整
        await asyncio.gather(*sched._build_tasks.values())

    asyncio.run(main())


def test_failed_registration_leaves_no_visible_build(db, monkeypatch):
    """第二步登记失败：报错、零残留、重启恢复不受影响。"""
    original = Database._execute
    state = {"step_inserts": 0}

    async def flaky(self, sql, params=()):
        if "INSERT INTO steps" in sql:
            state["step_inserts"] += 1
            if state["step_inserts"] == 2:  # “打包”步登记失败
                raise sqlite3.OperationalError("模拟登记故障")
        return await original(self, sql, params)

    monkeypatch.setattr(Database, "_execute", flaky)

    async def main():
        sched = Scheduler(db)
        with pytest.raises(sqlite3.OperationalError, match="模拟登记故障"):
            await sched.submit(_payload(name="compile-package", steps=2))

        # 查询接口零残留
        assert await db.list_builds() == []
        # 数据库底层也零残留：连 registering 行都没有（事务已回滚）
        assert await db.all("SELECT * FROM builds") == []
        assert await db.all("SELECT * FROM steps") == []
        # 调度器没有为它启动任何任务
        assert sched._build_tasks == {}
        # 重启恢复：不存在需要标记中断的构建
        report = await sched.recover()
        assert report["builds"] == 0
        assert report["purged_registering"] == 0

    asyncio.run(main())


def test_rollback_does_not_take_concurrent_writes_down(db, monkeypatch):
    """登记回滚不得带走其他协程的并发写（写锁互斥）。"""
    entered = asyncio.Event()
    release = asyncio.Event()
    original = Database._execute

    async def gate_then_fail(self, sql, params=()):
        if "INSERT INTO builds" in sql:
            entered.set()
            await release.wait()
            raise sqlite3.OperationalError("登记失败，触发回滚")
        return await original(self, sql, params)

    async def main():
        sched = Scheduler(db)
        # 直接落一行正常构建（不经调度器），作为并发写的目标
        bid_ok = await db.execute(
            "INSERT INTO builds (name, status, created_at) "
            "VALUES ('ok', 'pending', 'now')"
        )
        monkeypatch.setattr(Database, "_execute", gate_then_fail)

        submit_task = asyncio.create_task(sched.submit(_payload(name="bad", steps=1)))
        await asyncio.wait_for(entered.wait(), timeout=5)

        # 登记事务打开期间，另一个协程发起写：必须等锁，不能混入事务
        write_task = asyncio.create_task(
            db.set_build_status(bid_ok, "running", "now", started=True)
        )
        await asyncio.sleep(0.1)  # 让写请求进入等锁状态
        release.set()
        with pytest.raises(sqlite3.OperationalError):
            await submit_task
        await write_task  # 登记回滚后，被挡住的写继续执行

        row = await db.get_build(bid_ok)
        assert row["status"] == "running"  # 并发写存活，未被回滚带走
        assert await db.all("SELECT * FROM builds WHERE status='registering'") == []

    asyncio.run(main())


def test_recover_purges_registering_leftovers(db):
    """防御路径：registering 残骸被清理且永不进入可见集合。"""

    async def main():
        # 手工构造"按事务设计不可能出现"的残骸，验证兜底
        bid = await db.execute(
            "INSERT INTO builds (name, status, created_at) "
            "VALUES ('broken', 'registering', 'now')"
        )
        await db.execute(
            "INSERT INTO steps (build_id, step_uid, command, depends_on, "
            "timeout_seconds, status) VALUES (?, 'compile', 'true', '[]', 5, 'pending')",
            (bid,),
        )
        # 清理前：查询接口已经看不到它
        assert await db.list_builds() == []
        assert await db.get_build(bid) is None

        sched = Scheduler(db)
        report = await sched.recover()
        assert report["purged_registering"] == 1
        assert report["builds"] == 0  # 不得被当作中断构建
        assert await db.all("SELECT * FROM builds") == []
        assert await db.all("SELECT * FROM steps") == []

    asyncio.run(main())


def test_api_returns_500_on_registration_failure(monkeypatch):
    """路由层：登记异常 -> 500 + 明确说明（配合零残留由上方测试保证）。"""
    from fastapi.testclient import TestClient

    import app.main as main_module

    async def boom(payload):
        raise sqlite3.OperationalError("模拟登记故障")

    monkeypatch.setattr(main_module.scheduler, "submit", boom)
    client = TestClient(main_module.app)
    r = client.post("/builds", json={
        "name": "compile-package",
        "steps": [
            {"id": "compile", "command": "true", "depends_on": [],
             "timeout_seconds": 5},
            {"id": "package", "command": "true", "depends_on": ["compile"],
             "timeout_seconds": 5},
        ],
    })
    assert r.status_code == 500
    assert "登记失败" in r.json()["detail"]


def test_concurrent_submits_are_each_complete(db):
    """并发登记：各自单事务序列化执行，全部完整可见。"""

    async def main():
        sched = Scheduler(db)
        bids = await asyncio.gather(
            *(sched.submit(_payload(name=f"b{i}", steps=4)) for i in range(5))
        )
        assert len(set(bids)) == 5
        for bid in bids:
            assert len(await db.get_steps(bid)) == 4
        await asyncio.gather(*sched._build_tasks.values())
        for bid in bids:
            row = await db.get_build(bid)
            assert row["status"] == "succeeded", row["detail"]

    asyncio.run(main())
