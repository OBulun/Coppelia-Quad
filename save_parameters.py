def save_parameters(Q, R, N, pid_params, filename="simulation_parameters.txt"):
    with open(filename, "w") as f:
        f.write("MPC Horizon (N): {}\n".format(N))
        f.write("PID Parameters:\n")
        for key, value in pid_params.items():
            f.write("  {}: {}\n".format(key, value))
        f.write("Q Matrix:\n{}\n".format(Q))
        f.write("R Matrix:\n{}\n".format(R))
    print("Parameters saved to", filename)