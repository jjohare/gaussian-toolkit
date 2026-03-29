# Docker Integration

The 3DGS stack is integrated into the agentic-workstation Docker image as **Phase 2.7** of the unified Dockerfile.

## Build Phase Summary

| Sub-phase | What | Time |
|-----------|------|------|
| 2.7a | System deps (cmake, ninja, eigen, ceres, etc.) | ~1 min |
| 2.7b | METIS 5.2.1 + GKlib from source | ~30 sec |
| 2.7c | COLMAP headless CUDA build | ~5 min |
| 2.7d | vcpkg bootstrap | ~30 sec |
| 2.7e | LichtFeld Studio build (91 vcpkg deps + app) | ~20 min |
| 2.7f | SplatReady plugin + Python deps | ~30 sec |
| 2.7g | CLI wrappers (lfs-mcp, video2splat) | ~5 sec |

Total additional build time: ~25-30 minutes (mostly vcpkg first-time compilation).

## Files Modified

- `multi-agent-docker/Dockerfile.unified` — Phase 2.7 block (+150 lines)
- `multi-agent-docker/docker-compose.unified.yml` — Environment variables
- `multi-agent-docker/config/lichtfeld-studio/` — Skill files for COPY

## Environment Variables

Added to both compose and `.zshrc`:

```
LICHTFELD_EXECUTABLE=/opt/lichtfeld-src/build/LichtFeld-Studio
LICHTFELD_MCP_ENDPOINT=http://127.0.0.1:45677/mcp
VCPKG_ROOT=/opt/vcpkg
LD_LIBRARY_PATH=/opt/lichtfeld-src/build:...
```

## Installed Paths (Inside Container)

| Component | Path |
|-----------|------|
| LichtFeld Studio binary | `/opt/lichtfeld-src/build/LichtFeld-Studio` |
| LichtFeld symlink | `/usr/local/bin/lichtfeld-studio` |
| COLMAP binary | `/usr/local/bin/colmap` |
| vcpkg | `/opt/vcpkg` |
| SplatReady plugin | `~/.lichtfeld/plugins/splat_ready/` |
| lfs-mcp CLI | `/usr/local/bin/lfs-mcp` |
| video2splat CLI | `/usr/local/bin/video2splat` |
| Skill docs | `~/.claude/skills/lichtfeld-studio/` |
| METIS static lib | `/usr/local/lib/libmetis.a` |
| GKlib static lib | `/usr/local/lib/libGKlib.a` |

## GPU Requirements

The Docker container requires:
- `runtime: nvidia` in compose
- `NVIDIA_DRIVER_CAPABILITIES: compute,utility,graphics,display`
- CUDA 12.0+ driver on host
- NVIDIA Container Toolkit installed

## MCP Server Registration

The LichtFeld MCP bridge is registered in `~/.claude.json`:

```json
{
  "mcpServers": {
    "lichtfeld": {
      "type": "stdio",
      "command": "python3",
      "args": ["/opt/lichtfeld-src/scripts/lichtfeld_mcp_bridge.py"],
      "env": {
        "LICHTFELD_EXECUTABLE": "/opt/lichtfeld-src/build/LichtFeld-Studio",
        "LD_LIBRARY_PATH": "/opt/lichtfeld-src/build"
      }
    }
  }
}
```

## RuVector Memory

The 3DGS stack knowledge is stored in the external RuVector PostgreSQL memory:

- **Project**: `gaussians` (ID 100)
- **Namespace**: `gaussians`
- **Entries**: 11 memory entries with 384-dim embeddings (tools, APIs, CLIs, workflows, build info)
- **Patterns**: 10 (5 workflow, 2 build, 3 troubleshooting)
- **Knowledge Graph**: 13 nodes, 17 edges in the `gaussians` graph
- **IMPORTANT**: Always use `mcp__claude-flow__memory_store` — never raw SQL INSERT. Raw SQL bypasses the embedding pipeline and entries become invisible to HNSW semantic search.
