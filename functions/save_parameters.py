def save_parameters(Q, R, N, pid_params_z=None,pid_params_x=None,pid_params_y=None, filename="simulation_parameters.txt"):
    with open(filename, "w") as f:
        f.write("MPC Horizon (N): {}\n".format(N))
        f.write("PID Parameters :\n")
        if pid_params_z is not None:
            f.write("  PID Z: ")
            for key, value in pid_params_z.items():
                f.write("  {}: {}".format(key, value))
            f.write("\n")
        if pid_params_x is not None:    
            f.write("  PID X: ")
            for key, value in pid_params_x.items():
                f.write("  {}: {}".format(key, value))   
            f.write("\n")
        if pid_params_y is not None:
            f.write("  PID Y: ") 
            for key, value in pid_params_y.items():
                f.write("  {}: {}".format(key, value))
            f.write("\n")
        f.write("Q Matrix:\n{}\n".format(Q))
        f.write("R Matrix:\n{}\n".format(R))
    print("Parameters saved to", filename)