import subprocess as sub

def run(p):
    p = sub.Popen(['./runRobot.sh', str(p[0]), str(p[1]), str(p[2])],
            stdout=sub.PIPE, stderr=sub.PIPE)
    output, errors = p.communicate()
    return float(output)

def twiddle(tol=0.2): 
    p = [0, 0, 0]
    dp = [1, 1, 1]
    best_err = run(p)
    err = best_err
    # twiddle loop here
    nrun = 0
    while sum(dp) > tol:
        print(nrun, "----------------------------------------------------------------")
        print(nrun, "CURRENT BEST param =", p)
        print(nrun, "            New dp =", dp)
        print(nrun, "          BEST ERR =",  best_err)
        for i in range(len(p)):
            p[i] += dp[i]
            print(nrun, "----------------------------------------------------------------")
            print(nrun, "Trying +dp =", dp)
            print(nrun, "     param =", p)
            err = run(p)
            if err < best_err:
                best_err = err
                dp[i] *= 1.1
                print(nrun, "BEST, ERR =", err)
            else:
                p[i] -= 2*dp[i]
                print(nrun, "ERR =", err)
                print(nrun, "----------------------------------------------------------------")
                print(nrun, "Trying -dp =", dp)
                print(nrun, "     param =", p)
                err = run(p)
                if err < best_err:
                    best_err = err
                    dp[i] *= 1.1
                    print(nrun, "BEST, ERR =", err)
                else:
                    print(nrun, "ERR =", err)
                    p[i] += dp[i]
                    dp[i] *= 0.9
        nrun += 1
    return p, best_err

params, err = twiddle()
# params = [0.21, 0.005, 3.05]
# err = run(params)
print("Final twiddle error = {}".format(err))
print("Final twiddle params = {}".format(params))
