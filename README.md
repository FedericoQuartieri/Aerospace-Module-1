# Navier-Stokes-Brinkman equation solver

## Makefile instructions

### Build main executable
```bash
make
```

### Run main executable
```bash
make run
```

### Build profiling version (Mac)
```bash
make profile
```

### Build and run all tests
```bash
make test
```

### Run one specific test
```bash
make run-test_manufactured
make run-test_paper_manufactured
make run-test_zero_pressure_manufactured
```

### Build and run profiling tests
```bash
make test-profile
```

### Run one specific profiling test
```bash
make run-profile-test_paper_manufactured
```

### Clean build files
```bash
make clean
```

## Convergence study (Python)

Before running `test_convergence`, check that `test_convergence.c` contains the manufactured solution and parameters you want to use.

### Usage
```bash
python run_convergence_study.py <name>
```
