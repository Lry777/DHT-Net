import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import OrderedDict
import numpy as np

font={'family':'Times New Roman',
     # 'style':'italic',
    'weight':'normal',
      'color':'black',
      'size':34
}

def init_colors():
    return ['red','blue','goldenrod', 'green',  'black' , 'purple', 'gray', 'yellow']

def init_colors_swin():
    return ['green', 'orangered' ,'firebrick',  'goldenrod']

def init_colors_seg():
    return ['blue', 'firebrick',  'goldenrod' , 'deeppink' ] # ['blue', 'orangered' ,'firebrick',  'glod' , 'lightsteelblue' ]


def show_graph(data, ax, save_png_name=None, colors=init_colors()):
    """
    绘制折线图
    :param data: 数据格式：{label:{X:Y}, label:{X:Y}...}
    :param save_png_name:保存的图片的名字
    :param colors: 颜色列表
    :return:
        None
    """

    # plt.figure(figsize=(14, 6))
    plts = []
    labels = []
    for index, label in enumerate(data.keys()):
        if index == 1:
            linestyle = '-'
            marker = 'o'
        else:
            linestyle = ':'
            marker = 'o'
        if label == 'rotate':
            continue
        color = colors[index]
        X = data.get(label).keys()
        Y = [data.get(label).get(x) for x in X]
        temp, = ax.plot(X, Y,linewidth=4.0, linestyle=linestyle, color=color, label=label, marker=marker)
        plt.tick_params(labelsize=28)
        plt.ylabel('Validation loss', fontdict=font)
        plt.xlabel('Epoch', fontdict=font)
        plts.append(temp)
        labels.append(label)
    font1 = {'family' : 'Times New Roman', 'size' : 30}
    plt.legend(handles=plts, labels=labels, prop=font1)
    # plt.show()
    # if save_png_name is not None:
    #     plt.savefig(save_png_name)


if __name__ == "__main__":
    plt.figure(figsize=(14, 8))
    # font2 = {'family' : 'Times New Roman', 'size' : 18}
    # 将整个figure分成两行四列
    ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=2)

    ########################
    data_ade_160k = OrderedDict()
    a = 1.0
    # a1 = 50.76/49.79
    # seg_ade = {1.6:43.32*a, 3.2:46.9*a, 4.8:47.84*a, 6.4:49.98*a, 8.0:50.12*a, 9.6:50.46*a, 11.2:50.47*a, 12.8:50.32*a, 14.4:50.6*a, 16.0:50.64*a}
    # swin_ade = {1.6:48.15*a1, 3.2:48.64*a1, 4.8:49.87*a1, 6.4:50.44*a1, 8.0:49.15*a1, 9.6:49.52*a1, 11.2:49.62*a1, 12.8:49.82*a1, 14.4:49.54*a1, 16.0:49.79*a1}
    # ours_ade = {1.6:45.49*a,3.2:47.42*a, 4.8:48.12*a, 6.4:49.69*a, 8.0:49.81*a, 9.6:50.4*a, 11.2:50.46*a, 12.8:50.66*a, 14.4:51*a, 16.0:51.18*a}
    Backbone00 = {0:0.0431+a, 2:-0.0969+a, 4:-0.1868+a, 6:-0.2130+a, 8:-0.2865+a, 10:-0.3429+a, 12:-0.3154+a, 14:-0.2941+a, 16:-0.3938+a, 18:-0.3686+a, 20:-0.3490+a}
    Backbone10 = {0:-0.0487+a, 2:-0.1615+a, 4:-0.3381+a, 6:-0.2916+a, 8:-0.3514+a, 10:-0.3627+a, 12:-0.2973+a, 14:-0.4025+a, 16:-0.4702+a, 18:-0.4155+a, 20:-0.4091+a}

    data_ade_160k['CNN'] = Backbone00
    data_ade_160k['DPT'] = Backbone10
    #output_path_ade_160k = '/root/We_Network/SegFormer-master/graph/ade_160k.png'

    show_graph(data_ade_160k,ax1)
    # ax1.set_title('a', fontdict=font2)

    plt.tight_layout()
    save_png_name = '/root/We_Network/dhtnet/text_garph/duibi.pdf'
    plt.savefig(save_png_name, bbox_inches='tight')
    plt.show()