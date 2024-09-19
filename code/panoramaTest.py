from load_data import *
import matplotlib.pyplot as plt
import numpy as np
import math

def main():
    print("load orientation data")
    dataset= "9"
    cfile = "../data/cam/cam" + dataset + ".p"
    ifile = "../data/imu/imuRaw" + dataset + ".p"
    vfile = "../data/vicon/viconRot" + dataset + ".p"
    
    ts = tic()
    camd = read_data(cfile)
    print("------------- camd -------------")
    print(camd)

    imud = read_data(ifile)
    print("------------- imud -------------")

    vicd = read_data(vfile)
    print("------------- vicd -------------")
  
    toc(ts,"Data import")
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
        
        while (index + 1 < len(vicd["ts"][0])) and camd["ts"][0][i] >= vicd["ts"][0][index+1]:
            index += 1

        latest[camd["ts"][0][i]] = index
        # print(str(camd["ts"][0][i]) + " > " + str(vicd["ts"][0][index]))

    T_I_C = np.array([np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0.1]), np.array([0, 0, 0, 1])]) # image to camera
    
    H  = 1080
    V = 3000
    panorama = np.zeros((V, H, 3))
    print("camd: ", camd["ts"][0])
    print("vicd: ", vicd["ts"][0])

    for i in range(0, len(camd["ts"][0]), 18):
        ind = latest[camd["ts"][0][i]]
        R_v = vicd["rots"][:, :, ind]

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

                #print("x = " + str(x) + " y = " + str(y))
                # print("theta = " + str(theta) + " psi = " + str(psi))
                # print("cartx = " + str(cart_x) + " carty = " + str(cart_y))


                # s_c = np.array([np.array([cart_x]), np.array([cart_y]), np.array([cart_z])])
                s_c = np.array([np.array([cart_z]), np.array([-1*cart_x]), np.array([cart_y])])
                # s_c = np.array([np.array([-1*cart_y]), np.array([cart_z]), np.array([cart_x])])

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
                #print("xp = " + str(xp) + " yp = " + str(yp) + " zp = " + str(zp))
                #print("theta p = " + str(theta_prime) + " psi p = " + str(psi_prime))
                
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


if __name__ == "__main__":
    main()