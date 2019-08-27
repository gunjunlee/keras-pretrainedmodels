import kerasmodels as KM 

import shy
shy.err_hook()

if __name__ == '__main__':
    model = KM.shufflenet_v2_x0_5(pretrained=True)
    model = KM.shufflenet_v2_x1_0(pretrained=True)

    model = KM.squeezenet1_0(pretrained=True)
    model = KM.squeezenet1_1(pretrained=True)

    model = KM.mobilenet_v2(pretrained=True)

    model = KM.mnasnet0_5(pretrained=True)
    model = KM.mnasnet1_0(pretrained=True)
    
    try:
        model = KM.mnasnet0_75(pretrained=True)
    except ValueError:
        pass
    try:
        model = KM.mnasnet1_3(pretrained=True)
    except ValueError:
        pass
    try:
        model = KM.shufflenet_v2_x0_5(pretrained=True)
    except ValueError:
        pass
    try:
        model = KM.shufflenet_v2_x0_5(pretrained=True)
    except ValueError:
        pass