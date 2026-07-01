from __future__ import annotations

import sys


def main() -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "FastAPI web dependencies are required. Install them with:\n"
            "  python -m pip install -r requirements-web.txt\n"
            f"Original error: {exc}"
        ) from exc

    uvicorn.run("web_app.backend.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
