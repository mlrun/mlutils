import os

def get_vol_mount(
    v3io_home_key: str = "V3IO_HOME",
    pvc_params = { 
        "pvc_name"          : "nfsvol",
        "volume_name"       : "nfsvol",
        "volume_mount_path" :"/home/jovyan/data"}
):
    """get volume mount function
    
    checks whether v3io_home_key is defined in the env, if not
    will mount a pvc as defined in pvc_params.  The default 
    values here are those to be used when following the local
    open source setup of mlrun:
    https://github.com/mlrun/mlrun/blob/master/hack/local/README.md
    
    reused everywhere, candidate for mlrun
    
    :param v3io_home_key:    check for this env variable, if exists return `mount_v3io`,
                             else return `mount_pvc`
    :param pvc_params:       parameters passed on to `mount_pvc`
    """
    if v3io_home_key in list(os.environ):
        from mlrun import mount_v3io
        return mount_v3io()
    else:
        from mlrun.platforms import mount_pvc
        return mount_pvc(**pvc_params)
    
