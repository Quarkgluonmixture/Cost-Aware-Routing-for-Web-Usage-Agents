---
type: audit
ref: B-1848
title: Playwright driver-wedge hang
status: deferred
priority: P1
effort: 2-3h
phase: post-fire
blocker: fire 跑中 p79/ immutable
---

# B-1848 · Playwright driver-wedge hang

silent infinite block 绕过 operation timeout + runner M1 + watchdog (Gate3 cls B0 som
task190 production incident 2026-05-23, §280)。runner MainThread 阻塞 Playwright sync
事件循环 `select` 等 wedged node driver (py-spy 定位); Playwright op-timeout 是 driver 侧
强制 (driver wedge 同死) + python 客户端无 IPC wall-clock deadline + M1 需 exception
(silent block 不抛) → 永久 hang, watchdog 仅 alert-only。

Fix: Playwright driver IPC / sync page-op 加**客户端 wall-clock deadline** (SIGALRM 或
watchdog-thread → page/context.close 超时) → wedge raise 进 runner M1 而非 silent block;
一并 reap B0 proxy CLOSE-WAIT pool 连接 (次要)。deferred post-fire (fire 跑中 `p79/` immutable)。
