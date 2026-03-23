import numpy as np
import matplotlib.pyplot as plt 


def gradient_descent(f_prime, x0, alpha, N, small):
    '''
    The Gradient descent function finds the min imum of any function 

    Parameters:
    
    f_prime: the derivative of the function we want to minimize 
    x0: initial guess
    alpha: learning code variable 
    N: max iteration 
    small: cap for the function to stop running

    Returns:

    x1: the approximation of the minimum 
    i + 1: number of iteration if < N
    N: if number of iteration = N
    aplpha 
    f_prime(x1)

    '''
    
    x0 = np.array(x0, dtype=float)

    for i in range(N):
        x1 = x0 - (alpha * f_prime(x0))

        # use vector norm for stopping condition
        if np.linalg.norm(x1 - x0) < small:
            return x1, i + 1, f_prime(x1)

        x0 = x1

    return x1, N, alpha

def TestGradientDescent():
    print(" Testing Gradient Descent Method...\n")

    # Basic function: f(x,y) = x^2 + y^2
    f_prime = lambda v: 2 * np.array(v)
    true_min = np.array([0.0, 0.0])

    start_points = [
        np.array([5.0, -3.0]),
        np.array([-4.0, 2.0]),
        np.array([3.0, 6.0])
    ]

    alpha = 0.1
    max_iter = 40
    small = 1e-6
    found_points = []

    plt.figure(figsize=(8, 6))
    print("Calculating convergence orders for Gradient Descent...\n")

    for x0 in start_points:
        x = np.array(x0, dtype=float)
        errors = []

        for _ in range(max_iter):
            err = np.linalg.norm(x - true_min)
            errors.append(err)

            x_new = x - alpha * f_prime(x)
            if np.linalg.norm(x_new - x) < small:
                break
            x = x_new

        found_points.append(x)

        # Estimate convergence order
        if len(errors) >= 3:
            log_e = np.log(errors[:-1])
            log_e_next = np.log(errors[1:])
            slope, _ = np.polyfit(log_e, log_e_next, 1)

            print(f"  From x0={x0} → min≈{x} (true {true_min}), order ≈ {slope:.3f}")
            plt.plot(log_e, log_e_next, 'o-', label=f"x0={x0}, order={slope:.2f}")

    plt.title("Gradient Descent Convergence Order")
    plt.xlabel("log||error_k||")
    plt.ylabel("log||error_{k+1}||")
    plt.grid(True)
    plt.legend()
    plt.savefig("gradient_descent_convergence.png")

    print("\nGradient Descent plot saved as 'gradient_descent_convergence.png'")
    print(f"Gradient Descent points found: {found_points}\n")

if __name__ == "__main__":
    print("\n--- Testing Gradient Descent Method ---\n")

    try:
        TestGradientDescent()
        print("Gradient Descent successfully reached the minimum for all start points!")
    except Exception as e:
        print(f"\n Error: {e}")