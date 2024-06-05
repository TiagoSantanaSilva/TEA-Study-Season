import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
import math
from random import randint

# The number of trades we want to simulate

# BANCOR CONSTANTS
NUMBER_OF_TRADES = 10

# sqrt(Phigh) - sqrt(Plow)
A = 2
# sqrt(Plow)
B = 1
# Y intersect
Z = 300
# P0
P0 = B*(A+B)

#START AMMS
Y0_1 = float(randint(0,Z)) #Z**2/(A**2*(X0_ad_1+(Z/(A*(A+B)))))-(B*Z/A)
X0_ad_1 = (Z**2/((A**2)*(Y0_1+((B*Z)/A))))-(Z/(A*(A+B))) # RANDOM X BETWEEN MIN AND MAX X => MAX X IS ARBITRARY

Y0_2 = float(randint(0,Z)) #Z**2/(A**2*(X0_ad_1+(Z/(A*(A+B)))))-(B*Z/A)
X0_ad_2 = (Z**2/((A**2)*(Y0_2+((B*Z)/A))))-(Z/(A*(A+B))) # RANDOM X BETWEEN MIN AND MAX X => MAX X IS ARBITRARY

xplot_ad_1_list = []
xplot_ad_2_list = []
yplot1_list = []
yplot2_list = []
dydxplot1_list = []
dydxplot2_list = []

dydx_0_1 = -(Z**2*(A+B)**2)/(X0_ad_1*A*(A+B)+Z)**2
dydx_0_2 = -(Z**2*(A+B)**2)/(X0_ad_2*A*(A+B)+Z)**2

# LISTS THAT WILL CONTAIN THE DATA
x_ad_1_trade=[]
y_1_trade=[]

x_ad_2_trade=[]
y_2_trade=[]

dydx_curve_2=[]
dydx_curve_1=[]

def trade(NUMBER_OF_TRADES, X0_ad_1,Y0_1,X0_ad_2,Y0_2,dydx_0_1,dydx_0_2,x_ad_1_trade,y_1_trade,x_ad_2_trade,y_2_trade,dydx_curve_2,dydx_curve_1,A,B,Z,yplot1_list, xplot_ad_1_list, dydxplot1_list, yplot2_list, xplot_ad_2_list, dydxplot2_list):
    x_ad_1 = X0_ad_1
    x_ad_2 = X0_ad_2
    
    y_1 = Y0_1
    y_2 = Y0_2
    
    dydx_1 = dydx_0_1
    dydx_2 = dydx_0_2
    
    Z1 = Z
    Z2 = Z
    
    Z1_list =[]
    Z2_list =[]
    
    deltay_list = []
    
    signal1 = []
    signal2 = []
    
    for i in range(NUMBER_OF_TRADES):
        nextx_ad_1 = nextx_ad_2 = nexty_2 = nexty_1 = 0
        while nexty_1<=0 or nexty_2 <= 0: 
            deltay = norm.rvs(loc=0, scale=300, size=1)
            deltay_list.append(deltay)
            
            nexty_1 = y_1 + deltay[0]
            if i==0 and nexty_1>Z1:
                continue
               
            if nexty_1<=0:
                continue
            elif nexty_1>Z1:
                Z1=nexty_1
                nextx_ad_1=0
                signal1.append(True)
            elif nexty_1>0 and nexty_1<Z1:
                nextx_ad_1=(Z1**2/((A**2)*(nexty_1+((B*Z1)/A))))-(Z1/(A*(A+B)))
                signal1.append(False)
           
            yplot_1 = np.linspace(0, Z1, 500)
            xplot_ad_1 = (Z1**2/((A**2)*(yplot_1+((B*Z1)/A))))-(Z1/(A*(A+B)))
            dydxplot1 = - (Z1**2*(A+B)**2)/(xplot_ad_1*A*(A+B)+Z1)**2
                
            yplot1_list.append(yplot_1)
            xplot_ad_1_list.append(xplot_ad_1)    
            dydxplot1_list.append(dydxplot1)
            Z1_list.append(Z1)  
               
            deltax1=nextx_ad_1-x_ad_1
            nexty_2 = y_2 + deltax1
            if nexty_2<=0:
                continue
            elif nexty_2>Z2:
                Z2=nexty_2
                nextx_ad_2=0
                signal2.append(True)
            elif nexty_2>0 and nexty_2<Z2:
                nextx_ad_2=(Z2**2/((A**2)*(nexty_2+((B*Z2)/A))))-(Z2/(A*(A+B)))
                signal2.append(False)
                
            yplot_2 = np.linspace(0, Z2, 500)
            xplot_ad_2 = (Z2**2/((A**2)*(yplot_2+((B*Z2)/A))))-(Z2/(A*(A+B)))
            dydxplot2 = -(Z2**2*(A+B)**2)/(xplot_ad_2*A*(A+B)+Z2)**2
                
            yplot2_list.append(yplot_2)
            xplot_ad_2_list.append(xplot_ad_2)    
            dydxplot2_list.append(dydxplot2)
            Z2_list.append(Z2)

            nextdydx_1 = - (Z1**2*(A+B)**2)/(nextx_ad_1*A*(A+B)+Z1)**2
            nextdydx_2 = - (Z2**2*(A+B)**2)/(nextx_ad_2*A*(A+B)+Z2)**2

        x_ad_1_trade.append([x_ad_1, nextx_ad_1])
        y_1_trade.append([y_1, nexty_1])
        
        x_ad_2_trade.append([x_ad_2, nextx_ad_2])
        y_2_trade.append([y_2, nexty_2])
        
        dydx_curve_1.append([dydx_1, nextdydx_1])
        dydx_curve_2.append([dydx_2, nextdydx_2])
        
        y_1=nexty_1 
        y_2=nexty_2
        x_ad_1=nextx_ad_1
        x_ad_2=nextx_ad_2
        dydx_1=nextdydx_1
        dydx_2=nextdydx_2
        
    return Z1_list, Z2_list, signal1, signal2, deltay_list, x_ad_1_trade, y_1_trade, x_ad_2_trade, y_2_trade, dydx_curve_1,dydx_curve_2, yplot1_list, xplot_ad_1_list, dydxplot1_list, yplot2_list, xplot_ad_2_list, dydxplot2_list

Z1_list, Z2_list,signal1,signal2,deltay_list, x_ad_1_trade,y_1_trade,x_ad_2_trade,y_2_trade,dydx_curve_1,dydx_curve_2, yplot1_list, xplot_ad_1_list, dydxplot1_list, yplot2_list, xplot_ad_2_list, dydxplot2_list = trade(NUMBER_OF_TRADES,X0_ad_1,Y0_1,X0_ad_2,Y0_2,dydx_0_1,dydx_0_2,x_ad_1_trade,y_1_trade,x_ad_2_trade,y_2_trade,dydx_curve_1,dydx_curve_2,A,B,Z, yplot1_list, xplot_ad_1_list, dydxplot1_list, yplot2_list, xplot_ad_2_list, dydxplot2_list)

# Creating subplots for each trade
for i in range(NUMBER_OF_TRADES):
    # Create a figure
    fig = plt.figure(figsize=(16, 10.5), facecolor='#eeeef2')

    # Define width and height ratios for each column and row respectively
    width_ratios = [1,1,1,1]
    height_ratios = [1, 9]

    # Create a GridSpec with 2x5 layout and specify width and height ratios
    gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=width_ratios, height_ratios=height_ratios)

    # Add subplots to the GridSpec and store them in a 2D list
    axes = []
    for row in range(2):
        for col in range(4):
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)
    
    axes[0].axis('off')
    axes[0].text(
        0.5,
        0.5, 
        f"ETH Bonding Curve\nTrade {i+1}", 
        ha='center', va='center', fontsize=12, fontstyle='italic',fontweight='bold'
    )
    if signal1[i]==1:
        axes[0].text(
            0.5,
            0, 
            f'Previous Curve: ETH = ({round(Z1_list[i-1]**2,1)} / ({round(A**2,1)} * (x+{round((Z1_list[i-1]/(A*(A+B))),1)}))) - {round(B*Z1_list[i-1]/A,1)}\nNew Curve: ETH = ({round(Z1_list[i]**2,1)} / ({round(A**2,1)} * (x+{round((Z1_list[i]/(A*(A+B))),1)}))) - {round(B*Z1_list[i]/A,1)}',
            ha='center', va='center', fontsize=8, fontstyle='italic',fontweight='light'
        )
    else:
        axes[0].text(
            0.5,
            0, 
            f'Curve: ETH = ({round(Z1_list[i]**2,1)} / ({round(A**2,1)} * (x+{round((Z1_list[i]/(A*(A+B))),1)}))) - {round(B*Z1_list[i]/A,1)}',
            ha='center', va='center', fontsize=8, fontstyle='italic',fontweight='light'
        )
    
    axes[1].axis('off')
    axes[1].text(
        0.5,
        0.5, 
        f"CC Bonding Curve\nTrade {i+1}", 
        ha='center', va='center', fontsize=12, fontstyle='italic',fontweight='bold'
    )
    if signal2[i]==1:
        axes[1].text(
            0.5,
            0, 
            f'Previous Curve: CC = ({round(Z2_list[i-1]**2,1)} / ({round(A**2,1)} * (y_1+{round((B*Z2_list[i-1]/A),1)}))) - {round(B*Z2_list[i-1]/A,1)}\nNew Curve: CC = ({round(Z2_list[i]**2,1)} / ({round(A**2,1)} * (y_1+{round((B*Z2_list[i]/A),1)}))) - {round(B*Z2_list[i]/A,1)}\n',
            ha='center', va='center', fontsize=8, fontstyle='italic',fontweight='light'
        )
    else:
        axes[1].text(
            0.5,
            0, 
            f'Curve: CC = ({round(Z2_list[i]**2,1)} / ({round(A**2,1)} * (y_1+{round((B*Z2_list[i]/A),1)}))) - {round(B*Z2_list[i]/A,1)}',
            ha='center', va='center', fontsize=8, fontstyle='italic',fontweight='light'
        )
        
    
    axes[2].axis('off')
    axes[2].text(
        0.5,
        0.5, 
        f"ETH Price Curve\nTrade {i+1}", 
        ha='center', va='center', fontsize=12, fontstyle='italic',fontweight='bold'
    )
    
    if signal1[i]==1:
        axes[2].text(
            0.5,
            0, 
            f"Previous Curve: dETHdx = - {round((Z1_list[i-1]**2)*(A+B)**2,1)} / (x*{round(A*(A*B),1)}+{Z1_list[i-1]})^2\nNew Curve: dETHdx = - {round((Z1_list[i]**2)*(A+B)**2,1)} / (x*{round(A*(A*B),1)}+{round(Z1_list[i],2)})^2", 
            ha='center', va='center', fontsize=8, fontstyle='italic',fontweight='light'
        )
    else:
        axes[2].text(
            0.5,
            0, 
            f"Curve: dETHdx = - {round((Z1_list[i]**2)*(A+B)**2,1)} / (x*{round(A*(A*B),1)}+{Z1_list[i]})^2", 
            ha='center', va='center', fontsize=8, fontstyle='italic',fontweight='light'
        )
    
    axes[3].axis('off')
    axes[3].text(
        0.5,
        0.5, 
        f"CC Price Curve\nTrade {i+1}", 
        ha='center', va='center', fontsize=12, fontstyle='italic',fontweight='bold'
    )
    if signal2[i]==1:
        axes[3].text(
            0.5,
            0, 
            f"Previous Curve: dCCdx = -(x*{round(A*(A+B),1)}+{Z2_list[i-1]})^2 / {round((Z2_list[i-1]**2)*(A+B)**2,1)}\nNew Curve: dCCdx = -(x*{round(A*(A+B),1)}+{Z2_list[i]})^2 / {round((Z2_list[i]**2)*(A+B)**2,1)}", 
            ha='center', va='center', fontsize=8, fontstyle='italic',fontweight='light' 
        )
    else:
        axes[3].text(
            0.5,
            0, 
            f"Curve: dCCdx = -(x*{round(A*(A+B),1)}+{Z2_list[i]})^2 / {round((Z2_list[i]**2)*(A+B)**2,1)}", 
            ha='center', va='center', fontsize=8, fontstyle='italic',fontweight='light' 
        )

    axes[4].plot(xplot_ad_1_list[i], yplot1_list[i], color='blue')
    if signal1[i]==1:
        axes[4].plot(xplot_ad_1_list[i-1], yplot1_list[i-1], color='grey')
    axes[4].scatter(
        [x_ad_1_trade[i][0], x_ad_1_trade[i][1]],
        [y_1_trade[i][0], y_1_trade[i][1]], 
        color=['red', 'blue'], 
        edgecolors=['black', 'black'],
        s=80
    ) 
    
    axes[4].set_xlabel("Non-dimensional")
    axes[4].set_ylabel("Token ETH Balance")
    axes[4].grid(True) 
    
    if signal1[i]==1:
        ax1_elements = [
            mlines.Line2D([], [], color='blue', markerfacecolor='red',markersize=1, label=f'ETH Curve'),
            mlines.Line2D([], [], color='grey', markerfacecolor='grey',markersize=1, label=f'Previous Curve'),
            mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='red', markersize=10, label=f'Before trade ({round(x_ad_1_trade[i][0],2)} , {round(y_1_trade[i][0],2)})'),
            mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='blue', markersize=10, label=f'After trade ({round(x_ad_1_trade[i][1],2)} , {round(y_1_trade[i][1],2)})'),
            mlines.Line2D([], [], color='none', markersize=0, label=f'ΔX = {round((x_ad_1_trade[i][1]-x_ad_1_trade[i][0]),2)}'),
            mlines.Line2D([], [], color='none', markersize=0, label=f'ΔY = {round((y_1_trade[i][1]-y_1_trade[i][0]),2)}'),
        ]
    else:
        ax1_elements = [
            mlines.Line2D([], [], color='blue', markerfacecolor='red',markersize=1, label=f'ETH Curve'),
            mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='red', markersize=10, label=f'Before trade ({round(x_ad_1_trade[i][0],2)} , {round(y_1_trade[i][0],2)})'),
            mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='blue', markersize=10, label=f'After trade ({round(x_ad_1_trade[i][1],2)} , {round(y_1_trade[i][1],2)})'),
            mlines.Line2D([], [], color='none', markersize=0, label=f'ΔX = {round((x_ad_1_trade[i][1]-x_ad_1_trade[i][0]),2)}'),
            mlines.Line2D([], [], color='none', markersize=0, label=f'ΔY = {round((y_1_trade[i][1]-y_1_trade[i][0]),2)}'),
        ]
    axes[4].legend(handles=ax1_elements, loc='best', shadow=True, fancybox=True, title='Legend', title_fontsize='7', fontsize='7', framealpha=1)
    
    # Adding arrows    
    arrow1 = FancyArrowPatch((x_ad_1_trade[i][0], y_1_trade[i][0]), (x_ad_1_trade[i][0], y_1_trade[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
    arrow2 = FancyArrowPatch((x_ad_1_trade[i][0], y_1_trade[i][1]), (x_ad_1_trade[i][1], y_1_trade[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
    arrow3 = FancyArrowPatch((x_ad_1_trade[i][0], y_1_trade[i][0]), (x_ad_1_trade[i][1], y_1_trade[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    axes[4].add_patch(arrow1)
    axes[4].add_patch(arrow2)
    axes[4].add_patch(arrow3)
    
    # axes[7].axis('off')
    axes[5].plot(xplot_ad_2_list[i], yplot2_list[i], color='red')
    if signal2[i]==1:
        axes[5].plot(xplot_ad_2_list[i-1], yplot2_list[i-1], color='grey')
    axes[5].scatter(
        [x_ad_2_trade[i][0], x_ad_2_trade[i][1]],
        [y_2_trade[i][0], y_2_trade[i][1]], 
        color=['red', 'blue'], 
        edgecolors=['black', 'black'],
        s=80
    )
    
    axes[5].set_xlabel("Non-dimensional")
    axes[5].set_ylabel("Token CC Balance")
    axes[5].grid(True) 
    
    if signal2[i]==1:
        ax2_elements = [
            mlines.Line2D([], [], color='red', markerfacecolor='red',markersize=1, label=f'CC Curve'),
            mlines.Line2D([], [], color='grey', markerfacecolor='grey',markersize=1, label=f'Previous Curve'),
            mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='red', markersize=10, label=f'Before Trade ({round(x_ad_2_trade[i][0],2)} , {round(y_2_trade[i][0],2)})'),
            mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='blue', markersize=10, label=f'After Trade ({round(x_ad_2_trade[i][1],2)} , {round(y_2_trade[i][1],2)})'),
            mlines.Line2D([], [], color='none', markersize=0, label=f'ΔX = {round((x_ad_2_trade[i][1]-x_ad_2_trade[i][0]),2)}'),
            mlines.Line2D([], [], color='none', markersize=0, label=f'ΔY = {round((y_2_trade[i][1]-y_2_trade[i][0]),2)}'),
        ]
    else: 
        ax2_elements = [
            mlines.Line2D([], [], color='red', markerfacecolor='red',markersize=1, label=f'CC Curve'),
            mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='red', markersize=10, label=f'Before Trade ({round(x_ad_2_trade[i][0],2)} , {round(y_2_trade[i][0],2)})'),
            mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='blue', markersize=10, label=f'After Trade ({round(x_ad_2_trade[i][1],2)} , {round(y_2_trade[i][1],2)})'),
            mlines.Line2D([], [], color='none', markersize=0, label=f'ΔX = {round((x_ad_2_trade[i][1]-x_ad_2_trade[i][0]),2)}'),
            mlines.Line2D([], [], color='none', markersize=0, label=f'ΔY = {round((y_2_trade[i][1]-y_2_trade[i][0]),2)}'),
        ]
    axes[5].legend(handles=ax2_elements, loc='best', shadow=True, fancybox=True, title='Legend', title_fontsize='7', fontsize='7', framealpha=1)
    
    # Adding arrows
    arrow4 = FancyArrowPatch((x_ad_2_trade[i][0], y_2_trade[i][0]), (x_ad_2_trade[i][0], y_2_trade[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
    arrow5 = FancyArrowPatch((x_ad_2_trade[i][0], y_2_trade[i][1]), (x_ad_2_trade[i][1], y_2_trade[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
    arrow6 = FancyArrowPatch((x_ad_2_trade[i][0], y_2_trade[i][0]), (x_ad_2_trade[i][1], y_2_trade[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    axes[5].add_patch(arrow4)
    axes[5].add_patch(arrow5)
    axes[5].add_patch(arrow6)
    
    
    axes[6].plot(xplot_ad_1_list[i], dydxplot1_list[i], color='blue')
    if signal1[i]==1:
        axes[6].plot(xplot_ad_1_list[i-1], dydxplot1_list[i-1], color='grey')
    axes[6].scatter(
        [x_ad_1_trade[i][0],x_ad_1_trade[i][1]],
        [dydx_curve_1[i][0], dydx_curve_1[i][1]], 
        color=['red', 'blue'], 
        edgecolors=['black', 'black'],
        s=80
    )
    
    if signal1[i]==1:
        ax3_elements = [
            mlines.Line2D([], [], color='blue', markersize=10, label='dETHdx Curve'),
            mlines.Line2D([], [], color='grey', markerfacecolor='grey',markersize=1, label=f'Previous Curve'),
            mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='red', markeredgecolor='black', markersize=10, label=f'Before Trade ({round(x_ad_1_trade[i][0],2)} , {round(dydx_curve_1[i][0],2)})'),
            mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='blue', markeredgecolor='black',markersize=10, label=f'After Trade ({round(x_ad_1_trade[i][1],2)} , {round(dydx_curve_1[i][1],2)})'),
            mlines.Line2D([], [], color='#afaffd', marker ='s', markersize=10, label= f'|ΔETH|={abs(round((y_1_trade[i][1]-y_1_trade[i][0]),2))}'), 
        ]
    else: 
        ax3_elements = [
            mlines.Line2D([], [], color='blue', markersize=10, label='dETHdx Curve'),
            mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='red', markeredgecolor='black', markersize=10, label=f'Before Trade ({round(x_ad_1_trade[i][0],2)} , {round(dydx_curve_1[i][0],2)})'),
            mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='blue', markeredgecolor='black',markersize=10, label=f'After Trade ({round(x_ad_1_trade[i][1],2)} , {round(dydx_curve_1[i][1],2)})'),
            mlines.Line2D([], [], color='#afaffd', marker ='s', markersize=10, label= f'|ΔETH|={abs(round((y_1_trade[i][1]-y_1_trade[i][0]),2))}'), 
        ]
    axes[6].legend(handles=ax3_elements, loc='best', shadow=True, fancybox=True, title='Legend', title_fontsize='7', fontsize='7', framealpha=1)

    axes[6].set_xlabel("Non-dimensional")
    axes[6].set_ylabel("dETH / dCC")
    axes[6].grid(True) 
    
    sectionx = np.linspace(x_ad_1_trade[i][0], x_ad_1_trade[i][1], 500)
    section_dydx = -(Z1_list[i]**2*(A+B)**2)/(sectionx*A*(A+B)+Z1_list[i])**2
    axes[6].fill_between(sectionx, section_dydx, alpha=0.5, color ='#afaffd')
    
    # Adding arrows
    arrow7 = FancyArrowPatch((x_ad_1_trade[i][0], dydx_curve_1[i][0]), (x_ad_1_trade[i][0], dydx_curve_1[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
    arrow8 = FancyArrowPatch((x_ad_1_trade[i][0], dydx_curve_1[i][1]), (x_ad_1_trade[i][1], dydx_curve_1[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
    arrow9 = FancyArrowPatch((x_ad_1_trade[i][0], dydx_curve_1[i][0]), (x_ad_1_trade[i][1], dydx_curve_1[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    axes[6].add_patch(arrow7)
    axes[6].add_patch(arrow8)
    axes[6].add_patch(arrow9)
    
    axes[7].plot(xplot_ad_2_list[i], dydxplot2_list[i], color='red')
    if signal2[i]==1:
        axes[7].plot(xplot_ad_2_list[i-1], dydxplot2_list[i-1], color='grey')
    axes[7].scatter(
        [x_ad_2_trade[i][0],x_ad_2_trade[i][1]],
        [dydx_curve_2[i][0], dydx_curve_2[i][1]], 
        color=['red', 'blue'], 
        edgecolors=['black', 'black'],
        s=80
    )
    
    if signal2[i]==1:
        ax4_elements = [
            mlines.Line2D([], [], color='red', markersize=10, label='dCCdx Curve'),
            mlines.Line2D([], [], color='grey', markerfacecolor='grey',markersize=1, label=f'Previous Curve'),
            mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='red', markeredgecolor='black', markersize=10, label=f'Before Trade ({round(x_ad_2_trade[i][0],2)} , {round(dydx_curve_2[i][0],2)})'),
            mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='blue', markeredgecolor='black',markersize=10, label=f'After Trade ({round(x_ad_2_trade[i][1],2)} , {round(dydx_curve_2[i][1],2)})'),
            mlines.Line2D([], [], color='#fdbcbc', marker ='s', markersize=10, label= f'|ΔETH|={abs(round((y_2_trade[i][1]-y_2_trade[i][0]),2))}'), 
        ]
    else:
        ax4_elements = [
            mlines.Line2D([], [], color='red', markersize=10, label='dCCdx Curve'),
            mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='blue', markeredgecolor='black',markersize=10, label=f'After Trade ({round(x_ad_2_trade[i][1],2)} , {round(dydx_curve_2[i][1],2)})'),
            mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='red', markeredgecolor='black', markersize=10, label=f'Before Trade ({round(x_ad_2_trade[i][0],2)} , {round(dydx_curve_2[i][0],2)})'),
            mlines.Line2D([], [], color='#fdbcbc', marker ='s', markersize=10, label= f'|ΔCC|={abs(round((y_2_trade[i][1]-y_2_trade[i][0]),2))}'),  
        ]
    axes[7].legend(handles=ax4_elements, loc='best', shadow=True, fancybox=True, title='Legend', title_fontsize='7', fontsize='7', framealpha=1)

    axes[7].set_xlabel("Non-dimensional")
    axes[7].set_ylabel("dCC / dETH")
    axes[7].grid(True) 
    
    # Integral Area
    sectionx = np.linspace(x_ad_2_trade[i][0], x_ad_2_trade[i][1], 500)
    section_dydx = -(Z2_list[i]**2*(A+B)**2)/(sectionx*A*(A+B)+Z2_list[i])**2
    axes[7].fill_between(sectionx, section_dydx, alpha=0.5, color='#fdbcbc')
     
    # Adding arrows
    arrow10 = FancyArrowPatch((x_ad_2_trade[i][0], dydx_curve_2[i][0]), (x_ad_2_trade[i][0], dydx_curve_2[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
    arrow11 = FancyArrowPatch((x_ad_2_trade[i][0], dydx_curve_2[i][1]), (x_ad_2_trade[i][1], dydx_curve_2[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
    arrow12 = FancyArrowPatch((x_ad_2_trade[i][0], dydx_curve_2[i][0]), (x_ad_2_trade[i][1], dydx_curve_2[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    axes[7].add_patch(arrow10)
    axes[7].add_patch(arrow11)
    axes[7].add_patch(arrow12)
    
    # # First subplot with the actual trades
    plt.tight_layout()
    plt.show()