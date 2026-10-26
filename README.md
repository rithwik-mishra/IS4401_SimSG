To run backend for testing:

```
python -m pytest tests/test_simulation_context_route.py
```

To run the interactive swagger docs UI at https://127.0.0.1:8000/docs:

```
python -m uvicorn app.main:app --reload
```

