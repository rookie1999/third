### start VR in steam

### start ros pipe
'''cd /home/lumos/data_collection_11.14/start_process'''
'''bash ./unified_launcher.sh '''
### start data collection scripts
'''cd /home/lumos/data_collection_11.14/data_collector_opt'''
'''bash multi_session_collector_buffered.sh '''


'''conda activate piper'''

# replay
change the DATA_ROOT in config.yaml for traj data from umi for replay
'''bash ./piper/bash_scripts/can_activate.sh'''
 

### multi-remote-control
'''python remote_control.py'''

### multi-replay
'''python replay.py'''

configs are in the config.yaml


