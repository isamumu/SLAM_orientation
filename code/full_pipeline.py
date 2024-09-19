from load_data import *
# from rotplot import *
import autograd.numpy as np
from autograd import grad #, jit
#from jax.config import config
import math
import transforms3d 
import matplotlib.pyplot as plt

def cost(q, taus, omegas, alphas):
    t1 = 0
    t2 = 0
    T = len(q)

    for t in range(1, T):
        q_t = q[t]
        # t1
        if t < T-1:
            q_next = q[t+1]
            pert = np.array([0.000001, 0.000001, 0.000001, 0.000001]) # perturbation to prevent nan

            inv_term = quat_inv(q_next)
            f_term = quat_motion(q_t, taus[t-1], omegas[t-1])
            mult_term = quat_mult(inv_term, f_term)
            log_term = quat_log(mult_term + pert)
            t1 += quat_norm(2*log_term)**2
        
        # t2
        h_term = quat_observation(q_t)
        # # print("hterm = ", h_term[1:])
        # print("diff: " +  str(alphas[t-1]) + " minus " + str(h_term[1:]))
        # print("--> ", alphas[t-1] - h_term[1:])
        t2 += np.linalg.norm(alphas[t-1] - h_term[1:])**2

    t1 *= 0.5
    t2 *= 0.5

    print("observation loss: ", t2)
    print("motion loss: ", t1)

    return t1 + t2

def gradient(q_train, alpha, num_epochs, taus, omegas, alphas):
    q = np.array(q_train)
    q_prev = np.array(q_train) # array of quaternions
    grad_f = grad(cost)

    loss = []

    for i in range(num_epochs):
        print("epoch #", i)
        gradi = grad_f(q_prev, taus, omegas, alphas)
        # print("gradi: ", gradi)
        alphagrad = np.array(alpha*gradi)
    
        new_q = np.subtract(q_prev, alphagrad)
        row_norm = np.linalg.norm(new_q, axis=1, keepdims=True)# new_q.sum(axis=1)
        q = new_q / row_norm
        c = cost(q, taus, omegas, alphas) # print out loss
        loss.append(c)
        q_prev = q
    
    plt.plot(loss, color='r')
    plt.xlabel("# Epochs")
    plt.ylabel("Loss")

    plt.show()

    return q

def quat_log(q):
    qv = q[1:]
    qs = q[0]

    q_zero = np.array([0])
    q_test = np.concatenate((q_zero, qv))
    norm_qv = quat_norm(q_test)

    if norm_qv < 0.001:
        return np.array([np.log(abs(qs)), 0, 0, 0])

    norm_q = quat_norm(q)

    first = np.array([np.log(norm_q)])
    second = (qv/(norm_qv)) * np.arccos(qs/norm_q)

    return np.concatenate((first, second))

def quat_observation(q):
    # print(q)
    g = np.array([0,0,0,-9.81])
    inter = quat_mult(g, q)
    at = quat_mult(quat_inv(q), inter)
    # print(at)
    return at

def quat_norm(q):
    qs = q[0]
    qv = q[1:]

    return np.sqrt(qs**2 + np.dot(qv,qv))

def quat_inv(q):
    q_conj = quat_conj(q)
    den = quat_norm(q) 

    return q_conj/(den**2)

def quat_conj(q):
    qv = q[1:]
    qs = q[0]

    qs_v = np.array([qs])
    q_conj = np.concatenate((qs_v, -1*qv))

    return q_conj

def hatMap(x):
    return np.array([[0, -1*x[2], x[1]], [x[2], 0, -1*x[0]], [-1*x[1], x[0], 0]])

def quat2Rot(q):
    qv = np.array([q[1:]])
    qvHat = hatMap(q[1:])
    
    qs = q[0]
    col = qv.T

    I = np.identity(3)
    E = np.hstack((-1*col, qs*I + qvHat))
    G = np.hstack((-1*col, qs*I - qvHat))
    
    return np.dot(E, G.T)

def quat_mult(q, p):
    qs = q[0]
    ps = p[0]
    qv = q[1:]
    pv = p[1:]
    
    sol = np.array([qs*ps - np.dot(qv, pv)])
    sol2 = qs*pv + ps*qv + np.cross(qv, pv)
    return np.concatenate((sol, sol2))

def quat_exp(q):
    qv =q[1:]
    qs = q[0]

    norm_1 = np.array([0])
    norm_q = np.concatenate((norm_1, qv))

    if quat_norm(norm_q) < 0.001:
        return np.array([1.0, 0, 0, 0])

    sol = np.array([np.cos(quat_norm(norm_q))])

    sol2 = (qv/quat_norm(norm_q)) * np.sin(quat_norm(norm_q))
    return np.exp(qs)*np.concatenate((sol, sol2))

def quat_motion(qt, tau, wt):
    qv = (tau*wt) / float(2)

    zero = np.array([0])
    q = np.concatenate((zero, qv))
    expon = quat_exp(q)

    q_next = quat_mult(qt, expon)

    return q_next

def main():
    print("load orientation data")
    dataset="11"
    cfile = "../data/cam/cam" + dataset + ".p"
    ifile = "../data/imu/imuRaw" + dataset + ".p"
    # vfile = "../data/vicon/viconRot" + dataset + ".p"

    ts = tic()
    # TODO: uncomment for panorama
    camd = read_data(cfile)
    print("------------- camd -------------")
    # print(camd)
    imud = read_data(ifile)
    print("------------- imud -------------")
    print(imud)
    # vicd = read_data(vfile)
    
    toc(ts,"Data import")

    # accelerometer sensitivity -> 300 mV/g
    # gyro sensitivity (pitch/roll/yaw) -> 3.33 mV/deg/s
    sens_a = 300 / (-9.81) 
    sens_g = 3.33
    scale_a = 3300.0 / 1023.0 / sens_a
    scale_g = 3300.0 / (1023*sens_g) * math.pi / 180.0

    accum_ax = 0.0
    accum_ay = 0.0
    accum_az = 0.0
    accum_wx = 0.0
    accum_wy = 0.0
    accum_wz = 0.0

    # ================= calibration ================= # 
    for i in range(100, 400) :

        accum_ax += imud["vals"][0][i]
        accum_ay += imud["vals"][1][i]
        accum_az += imud["vals"][2][i]
        accum_wz += imud["vals"][3][i]
        accum_wx += imud["vals"][4][i]
        accum_wy += imud["vals"][5][i]

    bias_ax = accum_ax / 300.0 
    bias_ay = accum_ax / 300.0 
    bias_az = accum_az / 300.0
    bias_wx = accum_wx / 300.0
    bias_wy = accum_wy / 300.0 
    bias_wz = accum_wz / 300.0

    
    # ================= initial set of quaternions from IMU data ================= # 
    measured_rolls = []
    measured_pitches = []
    measured_yaws = []
    q_train = [] # store the set of quaternions

    q_prev = np.array([1.0, float(0), float(0), float(0)])
    q_train.append(q_prev)

    prev_stamp = float(imud["ts"][0][0])

    taus = []
    omegas = []
    alphas = []

    a1 = []
    a2 = []
    a3 = []

    a1m = []
    a2m = []
    a3m = []

    T = len(imud["vals"][3])
    for i in range(1, T):
        # obtain measured linear accelerations
        unbiased_ax = -1*(imud["vals"][0][i] - bias_ax) * scale_a
        unbiased_ay = -1*(imud["vals"][1][i] - bias_ay) * scale_a
        unbiased_az = (imud["vals"][2][i] - bias_az) * scale_a - 9.81 # TODO: verify that gravity units is negative

        #debug 
        a1.append(unbiased_ax)
        a2.append(unbiased_ay)
        a3.append(unbiased_az)

        v = np.array([unbiased_ax, unbiased_ay, unbiased_az])
        alphas.append(v) # mesaurement for cost function

        # obtain measured angular velocities
        unbiased_wz = (imud["vals"][3][i] - bias_wz) * scale_g
        unbiased_wx = (imud["vals"][4][i] - bias_wx) * scale_g
        unbiased_wy = (imud["vals"][5][i] - bias_wy) * scale_g 
        wt = np.array([unbiased_wx, unbiased_wy, unbiased_wz]) 
        omegas.append(wt) # measurement info for cost function

        # time difference computation
        tau = float(imud["ts"][0][i]) - prev_stamp
        prev_stamp = float(imud["ts"][0][i]) # update the prev stamp 
        taus.append(tau) # taus for cost
        
        # next quaternion estimation and Rotation matrix recovery
        q_next = quat_motion(q_prev, tau, wt)

        a = quat_observation(q_next)
        a1m.append(a[1])
        a2m.append(a[2])
        a3m.append(a[3])

        R = quat2Rot(q_next) # quaternion_rotation_matrix(q_next) 

        # obtain from R -> euler and store information
        az_m, ay_m, ax_m = transforms3d.euler.mat2euler(R, 'szyx')
        
        measured_rolls.append((ax_m*180)/math.pi)
        measured_pitches.append((ay_m*180)/math.pi)
        measured_yaws.append((az_m*180)/math.pi)
        q_train.append((q_next*180)/math.pi)

        q_prev = q_next # update next quaternion
    
    rolls_opt = []
    pitches_opt = []
    yaws_opt = []

    print("computing the gradient:")

    # TODO: uncomment later
    # 0.0008 works too
    q_new = gradient(q_train, 0.003, 10, taus, omegas, alphas)
    R_news = [0]
    for i in range(1, T):
        R = quat2Rot(q_new[i]) # quaternion_rotation_matrix(q_next) 

        R_news.append(R)

        # obtain from R -> euler and store information
        az_m, ay_m, ax_m = transforms3d.euler.mat2euler(R, 'szyx')
        rolls_opt.append((ax_m*180)/math.pi)
        pitches_opt.append((ay_m*180)/math.pi)
        yaws_opt.append((az_m*180)/math.pi)

    # ========================= construct panorama ========================= 
    # COMMENT OUT WHEN TESTING ORIENTATION ONLY
    T = len(camd["ts"][0])
    print("T = ", T)

    horView_deg = 60
    vertView_deg = 45

    # 320 × 240 RGB images
    x_center = 320 / 2
    y_center = 240 / 2

    tic_degree_h = (horView_deg) / 320
    tic_degree_v = (vertView_deg) / 240

    latest = {} # maps time in camd to index relating to latest time in vicon

    index = 0
    for i in range(len(camd["ts"][0])):
        
        while (index + 1 < len(imud["ts"][0])) and camd["ts"][0][i] >= imud["ts"][0][index+1]:
            index += 1

        latest[camd["ts"][0][i]] = index
        # print(str(camd["ts"][0][i]) + " > " + str(vicd["ts"][0][index]))

    T_I_C = np.array([np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0.1]), np.array([0, 0, 0, 1])]) # image to camera
    
    H  = 1080
    V = 5000
    panorama = np.zeros((V, H, 3))
    print("camd: ", camd["ts"][0])

    for i in range(0, len(camd["ts"][0]), 18):
        ind = latest[camd["ts"][0][i]]
        R_v = R_news[ind] # vicd["rots"][:, :, ind]

        zero = np.array([0])
        r1 = np.concatenate((R_v[0], zero))
        r2 = np.concatenate((R_v[1], zero))
        r3 = np.concatenate((R_v[2], zero))

        T_W_I = np.array([r1, r2, r3,np.array([0, 0, 0, 1])])

        # print(str(T_W_I) + " by " + str(T_I_C))
        T_W_C = np.matmul(T_W_I, T_I_C) # c -> w
        r1 = T_W_C[0][:3]
        r2 = T_W_C[1][:3]
        r3 = T_W_C[2][:3]
        R = np.array([r1, r2, r3])
        p = np.array([np.array(T_W_C[0][3]), np.array(T_W_C[1][3]), np.array(T_W_C[1][3])])

        mtx = camd["cam"][:,:,:,i]
        for x in range(320):
            for y in range(240):
                rgb_val = mtx[y][x]

                longitude = (x - x_center) * tic_degree_h # longitude / theta
                latitude = (y_center - y) * tic_degree_v # latitude / psi

                # print("theta p = " + str(theta) + " psi p = " + str(psi))
                theta = -1
                psi = -1

                if longitude >= 0:
                    theta = 90 - longitude
                else:
                    theta = 90 + abs(longitude)

                if latitude >= 0:
                    psi = 90 - latitude
                else: 
                    psi = 90 + abs(latitude)

                radius = H / (2 * math.pi)
                # cartesian conversion
                cart_x = np.sin((psi*math.pi)/180) * np.cos((theta*math.pi)/180)
                cart_y = np.cos((psi*math.pi)/180)
                cart_z = np.sin((theta*math.pi)/180) * np.sin((psi*math.pi)/180)

                cart_x = cart_x / cart_z
                cart_y = cart_y / cart_z
                cart_z = 1

                #s_c = np.array([np.array([cart_x]), np.array([cart_y]), np.array([cart_z])])
                s_c = np.array([np.array([cart_z]), np.array([-1*cart_x]), np.array([cart_y])])
                
                # rotate to world frame
                s_w = R.dot(s_c) + p
                # print("TWC: ", T_W_C)
                # print("sc = ", s_c)
                # print("sw = ", s_w)
                xpp = s_w[0][0]
                ypp = s_w[1][0]
                zpp = s_w[2][0]

                xp = -1*ypp
                yp = zpp
                zp = xpp

                radius = H / (2 * math.pi)

                # apply projection (https://medium.com/@suverov.dmitriy/how-to-convert-latitude-and-longitude-coordinates-into-pixel-offsets-8461093cb9f5)
                psi_prime = (np.arctan(np.sqrt(xp**2 + zp**2)/yp))

                theta_prime = (np.arctan(zp/(xp)))
                # print("xp = " + str(xp) + " yp = " + str(yp) + " zp = " + str(zp))
                # print("theta p = " + str(theta_prime) + " psi p = " + str(psi_prime))
                
                lat = psi_prime
                long = theta_prime 

                if psi_prime < 0:
                    lat = -1 * (math.pi/2 - abs(psi_prime))
                else:
                    lat = math.pi/2 - psi_prime

                if theta_prime < 0:
                    long = -1 * (math.pi/2 - abs(theta_prime))
                else: 
                    long = math.pi/2 - theta_prime

            
                x_new = int((long+math.pi) * radius)

                yFromEquator = radius * math.log(math.tan(math.pi / 4 + lat / 2))
                y_new = int(V / 2 - yFromEquator)
                
                panorama[y_new][x_new][:] = rgb_val

    plt.imshow(panorama.astype('uint8'))
    plt.show()
    # finish plotting panorama image

    # ================= plotting ================= # 
    plt.figure("roll", figsize=(20,5))
    plt.plot(measured_rolls, color='b')
    plt.plot(rolls_opt, color='g')
    plt.legend(['estimated', 'optimized'])
    plt.xlabel("Time")
    plt.ylabel("Angular Velocity")
    plt.figure("yaw", figsize=(20,5))
    plt.plot(measured_yaws, color='b')
    plt.plot(yaws_opt, color='g')
    plt.legend(['estimated', 'optimized'])
    plt.xlabel("Time")
    plt.ylabel("Angular Velocity")
    plt.figure("pitch", figsize=(20,5))
    plt.plot(measured_pitches, color='b')
    plt.plot(pitches_opt, color='g')
    plt.legend(['estimated', 'optimized'])
    plt.xlabel("Time")
    plt.ylabel("Angular Velocity")

    plt.ylim(-180, 180)
    plt.show()
    
if __name__ == "__main__":
    main()