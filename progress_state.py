import threading


_lock = threading.Lock()
_state = {
    "status": "idle",
    "uid": None,
    "stage": None,
    "detail": None,
    "progress_current": None,
    "progress_total": None,
    "progress_percent": None,
    "message": None,
}


def snapshot():
    with _lock:
        return dict(_state)


def reset():
    with _lock:
        _state.update(
            {
                "status": "idle",
                "uid": None,
                "stage": None,
                "detail": None,
                "progress_current": None,
                "progress_total": None,
                "progress_percent": None,
                "message": None,
            }
        )


def update(**updates):
    with _lock:
        _state.update(updates)


def set_stage(stage, detail=None, status="processing", current=None, total=None, percent=None, message=None):
    with _lock:
        _state.update(
            {
                "status": status,
                "stage": stage,
                "detail": detail,
                "progress_current": current,
                "progress_total": total,
                "progress_percent": percent,
                "message": message,
            }
        )
