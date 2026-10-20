"""完整链路在本机 Redis 未监听时，用 Docker 启动容器。"""

import subprocess

import tests.test_full_chain as full_chain


def test_run_command_sets_password_when_configured():
    command = full_chain.redis_run_command(6380, "secret")
    assert command[:8] == [
        "docker", "run", "-d", "--name", "trip-planner-redis", "-p", "6380:6379", "redis:7",
    ]
    assert command[-2:] == ["--requirepass", "secret"]


def test_ensure_redis_does_not_start_docker_when_port_is_open(monkeypatch):
    monkeypatch.setattr(full_chain, "_redis_target", lambda: ("127.0.0.1", 6379, ""))
    monkeypatch.setattr(full_chain, "_redis_reachable", lambda *args, **kwargs: True)
    called = []
    monkeypatch.setattr(full_chain, "subprocess", type("S", (), {"run": lambda *a, **k: called.append(a)})())
    assert full_chain.ensure_redis() == "already_running"
    assert called == []


def test_ensure_redis_leaves_remote_host_alone(monkeypatch):
    monkeypatch.setattr(full_chain, "_redis_target", lambda: ("redis.internal", 6379, ""))
    monkeypatch.setattr(full_chain, "_redis_reachable", lambda *args, **kwargs: False)
    assert full_chain.ensure_redis() == "remote_unreachable"


def test_ensure_redis_creates_container_when_local_port_is_closed(monkeypatch):
    seen = []

    def reachable(*args, **kwargs):
        return bool(seen)

    def run(cmd, **kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(full_chain, "_redis_target", lambda: ("127.0.0.1", 6379, ""))
    monkeypatch.setattr(full_chain, "_redis_reachable", reachable)
    monkeypatch.setattr(full_chain.shutil, "which", lambda name: "docker")
    assert full_chain.ensure_redis(run) == "started"
    assert seen[0][:3] == ["docker", "ps", "-aq"]
    assert seen[1][:3] == ["docker", "run", "-d"]
