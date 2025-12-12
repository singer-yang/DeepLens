# Run all tests
cd ./DeepLens
pytest test/ -v

# Run specific test file
pytest test/test_ray.py -v

# Run with coverage
pytest test/ --cov=deeplens --cov-report=term-missing