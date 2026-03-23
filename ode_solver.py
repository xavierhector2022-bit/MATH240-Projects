import numpy as np
import matplotlib.pyplot as plt 


def euler(f, t0, y0, t_end, h):
    """
    Solve the initial value problem y' = f(x, y)
    using Euler's method with step size h.

    Parameters:
        f (function): Function f(x, y) returning dy/dx
        t0 (float): Initial x value
        y0 (float): Initial y value
        t_end (float): Final x value
        h (float): Step size
       
    Returns:
        t (ndarray): Array of t values
        y (ndarray): Array of approximate y values
    """
    # Number of Steps 
    t = np.arange(t0, t_end + h, h)
    y = np.zeros((len(t), len(np.atleast_1d(y0))))
    y[0] = y0

    # Euler Method 
    for i in range(len(t)-1):
        y[i+1] = y[i] + h * f(t[i], y[i])

    return t, y
    

def RungeKutta(f, t0, y0, t_end, h):
    """
    Solve the initial value problem y' = f(x, y)
    using the Runge-Kutta method.

    Parameters:
        f (function): Function f(x, y) returning dy/dx
        t0 (float): Initial x value
        y0 (float): Initial y value
        t_end (float): Final x value
        h (float): Step size
        

    Returns:
        t (ndarray): Array of t values
        y (ndarray): Array of approximate y values
    """
    # Number of Steps 
    t = np.arange(t0, t_end + h, h)
    y = np.zeros((len(t), len(np.atleast_1d(y0))))
    y[0] = y0

    # RungeKutta Method
    for i in range(len(t)-1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h/2 * k1)
        k3 = f(t[i] + h/2, y[i] + h/2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i+1] = y[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return t, y


def TestEuler():
    print("\n--- Testing Euler Method ---\n")
    
    f = lambda t, y: y - t**2 + 1            # ODE
    exact = lambda t: (t + 1)**2 - 0.5*np.exp(t)   # Exact solution

    t0, y0, t_end = 0, 0.5, 2
    hs = [0.5, 0.25, 0.125, 0.0625]
    errors = []

    for h in hs:
        t, y = euler(f, t0, y0, t_end, h)
        error = abs(y[-1] - exact(t_end))
        errors.append(error)
        print(f"h = {h:<8} → y({t_end}) ≈ {y[-1][0]:.6f}, true = {exact(t_end):.6f}, error = {error[0]:.2e}")

    # Convergence order
    log_h = np.log(hs)
    log_err = np.log(np.array(errors).flatten())
    slope, _ = np.polyfit(log_h, log_err, 1)

    plt.figure(figsize=(7, 5))
    plt.plot(log_h, log_err, 'o-', label=f"Order ≈ {abs(slope):.2f}")
    plt.title("Euler Method Convergence Order")
    plt.xlabel("log(h)")
    plt.ylabel("log(Error)")
    plt.grid(True)
    plt.legend()
    plt.savefig("euler_convergence.png")
    plt.close()

    print(f"\nEuler plot saved as 'euler_convergence.png'")
    print(f"Estimated order of convergence ≈ {abs(slope):.2f}\n")


def TestRungeKutta():
    print("\n--- Testing Runge–Kutta 4th Order Method ---\n")
    
    f = lambda t, y: y - t**2 + 1
    exact = lambda t: (t + 1)**2 - 0.5*np.exp(t)

    t0, y0, t_end = 0, 0.5, 2
    hs = [0.5, 0.25, 0.125, 0.0625]
    errors = []

    for h in hs:
        t, y = RungeKutta(f, t0, y0, t_end, h)
        error = abs(y[-1] - exact(t_end))
        errors.append(error)
        print(f"h = {h:<8} → y({t_end}) ≈ {y[-1][0]:.6f}, true = {exact(t_end):.6f}, error = {error[0]:.2e}")

    # Convergence order
    log_h = np.log(hs)
    log_err = np.log(np.array(errors).flatten())
    slope, _ = np.polyfit(log_h, log_err, 1)

    plt.figure(figsize=(7, 5))
    plt.plot(log_h, log_err, 'o-', label=f"Order ≈ {abs(slope):.2f}")
    plt.title("Runge–Kutta 4th Order Convergence Order")
    plt.xlabel("log(h)")
    plt.ylabel("log(Error)")
    plt.grid(True)
    plt.legend()
    plt.savefig("runge_kutta_convergence.png")
    plt.close()

    print(f"\nRunge–Kutta plot saved as 'runge_kutta_convergence.png'")
    print(f"Estimated order of convergence ≈ {abs(slope):.2f}\n")


if __name__ == "__main__":
    print("\n--- Testing ODE Solvers ---\n")
    try:
        TestEuler()
        TestRungeKutta()
        print("Both Euler and Runge–Kutta tests completed successfully!\n")
    except Exception as e:
        print(f"\nError: {e}\n")