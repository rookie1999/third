import numpy as np

# calibrate

T_base2local = np.array([   [ 0         ,  0.        ,  1         ,  0.04  ],
                            [ 0.        ,  1.        ,  0.        ,  0.1        ],
                            [-1.        ,  0.        ,  0         ,  0.28  ],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]     ]) 
#X: 225114, Y: -13338, Z: 265780
T_base2local2 = np.array([   [  0         ,  0.        ,  1         ,  0.21  ],
                            [   0.        ,  1.        ,  0.        ,  -0.01        ],
                            [   -1        ,  0.        ,  0         ,  0.24  ],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]     ])  

T_base2local3 = np.array([  [   0         ,  1        ,          0  ,  0.1  ],
                            [   -1        ,    0      ,   0        ,  0.1        ],
                            [   0        ,   0.        ,  1        ,  0.4  ],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]     ])  
#replay speed
speed_rate = 1
DATA_ROOT = "~/data_collection_11.14/data_collector_opt"

# karmal filter param
dt_est=0.01,          # ~100 Hz
pos_std_meas=5,      # measurement noise: 1 mm
pos_std_acc=20.0,      # allow some acceleration
ori_alpha=0.7          # moderate smoothing

