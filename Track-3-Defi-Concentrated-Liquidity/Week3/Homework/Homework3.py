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
A = 3

S1_X0_1 = 100
S1_Y0_1  = 150

C1 = round(S1_X0_1 * S1_Y0_1 * A**2,2)

S1_X0_2  = 80
S1_Y0_2  = (S1_Y0_1/S1_X0_1)*((A**4)/(A-1)**4) * S1_X0_2

C2 = round(S1_X0_2 * S1_Y0_2 * A**2,2)

# UNISWAP CONSTANTS
L_1 = 500
L_2 = 2000
P_LOW_1 = 1
P_HIGH_1 = P_LOW_2 = 2
P_HIGH_2 = 3


# INITIALIZE SEPARATE UNISWAP FUNCTION
def initialize_separate_bonding_curves_uni(L,P_LOW,P_HIGH):
    Xmax=L*((1/math.sqrt(P_LOW))-(1/math.sqrt(P_HIGH)))
    Ymax=L*(math.sqrt(P_HIGH)-math.sqrt(P_LOW))
    xplot = np.linspace(0, Xmax, 500)
    yplot = (L**2 / (xplot+(L/math.sqrt(P_HIGH)))) - L*math.sqrt(P_LOW)
    dydxplot = - L**2 / (xplot+(L/math.sqrt(P_HIGH)))**2
    return xplot, yplot, dydxplot, Xmax, Ymax
    
# INITIALIZE SEPARATE UNISWAP BONDING CURVES
xplot1_uni, yplot1_uni, dydxplot1_uni, Xmax1_uni, Ymax1_uni = initialize_separate_bonding_curves_uni(L_1, P_LOW_1, P_HIGH_1) # UNI CURVE 1
xplot2_uni, yplot2_uni, dydxplot2_uni, Xmax2_uni, Ymax2_uni  = initialize_separate_bonding_curves_uni(L_2, P_LOW_2, P_HIGH_2) # UNI CURVE 2

# INITIALIZE JOINT UNISWAP FUNCTIONS
def initialize_joint_uniswap_curve_1(maxPoint1, maxPoint2, L, P_LOW, P_HIGH):
    xplot = np.linspace(maxPoint2, maxPoint2+maxPoint1, 500)
    yplot = (L**2 / ((xplot-maxPoint2)+(L/math.sqrt(P_HIGH)))) - L*math.sqrt(P_LOW)
    return xplot, yplot

def initialize_joint_uniswap_curve_2(Ymax1_bancor, Xmax2_bancor, L, P_LOW, P_HIGH):
    xplot = np.linspace(0, Xmax2_bancor, 500)
    yplot = (L**2 / (xplot+(L/math.sqrt(P_HIGH)))) - L*math.sqrt(P_LOW) + Ymax1_bancor
    dydxplot = - L**2 / ((xplot-Xmax2_bancor)+(L/math.sqrt(P_HIGH)))**2
    return xplot, yplot, dydxplot

# INITIALIZE JOINT UNISWAP BONDING CURVES
xplot3_uni, yplot3_uni = initialize_joint_uniswap_curve_1(Xmax1_uni, Xmax2_uni, L_1, P_LOW_1, P_HIGH_1)
xplot4_uni, yplot4_uni, dydxplot4_uni = initialize_joint_uniswap_curve_2(Ymax1_uni, Xmax2_uni, L_2, P_LOW_2, P_HIGH_2)
xplot5_uni = np.linspace(Xmax2_uni, Xmax1_uni+Xmax2_uni, 500)

#START AMMS
X0_reference_uni = float(randint(0,int(round(Xmax2_uni,0))-1)) # RANDOM X BETWEEN MAX AND MIN X
Y0_reference_uni = (L_2**2 / (X0_reference_uni+(L_2/math.sqrt(P_HIGH_2)))) - L_2*math.sqrt(P_LOW_2)

X=X0_reference_uni #StartX
Y=Y0_reference_uni  #StartY

# WHICH CURVE THE POINT BELONGS TO => CAN CHANGE ON STARTX AND STARTY
if round(((X+L_1/math.sqrt(P_HIGH_1))*(Y+L_1*math.sqrt(P_LOW_1))),2) == L_1**2:
    Xjoin = X+Xmax2_uni
    Yjoin = (L_1**2 / ((Xjoin-Xmax2_uni)+(L_1/math.sqrt(P_HIGH_1)))) - L_1*math.sqrt(P_LOW_1)
    dYdX = -L_1**2/(Xjoin+L_1/math.sqrt(P_HIGH_1))**2
elif round((X+L_2/math.sqrt(P_HIGH_2))*(Y+L_2*math.sqrt(P_LOW_2)),2) == L_2**2:
    Xjoin = X
    Yjoin = (L_2**2 / (Xjoin+(L_2/math.sqrt(P_HIGH_2)))) - L_2*math.sqrt(P_LOW_2) + Ymax1_uni
    dYdX = -L_2**2/(Xjoin+L_2/math.sqrt(P_HIGH_2))**2

# LISTS THAT WILL CONTAIN THE DATA
xtrade_uni = []
ytrade_uni = []

xjointrade_uni = []
yjointrade_uni = []

Xprice_curve_uni=[]
Yprice_curve_uni=[]

# TRADES IN UNISWAP
for _ in range(NUMBER_OF_TRADES):
    a=0
    b=0
    while a==0:
        aux_L1 = round((X+L_1/math.sqrt(P_HIGH_1))*(Y+L_1*math.sqrt(P_LOW_1)),2)
        aux_L2 = round((X+L_2/math.sqrt(P_HIGH_2))*(Y+L_2*math.sqrt(P_LOW_2)),2)
        deltax=norm.rvs(loc=0, scale=300, size=1)[0]
        
        if aux_L1 == L_1**2:
            if X+deltax>Xmax1_uni:
                continue
            elif X+deltax<0:
                deltax_aux = abs(deltax)-X
                nextx = Xmax2_uni-deltax_aux
                if nextx<0:
                    continue
                else:
                    nexty = (L_2**2 / (nextx+(L_2/math.sqrt(P_HIGH_2)))) - L_2*math.sqrt(P_LOW_2)
                    nextdydx = -L_2**2/(nextx+L_2/math.sqrt(P_HIGH_2))**2
                    a=1        
            else:
                nextx = X+deltax
                nexty = (L_1**2 / (nextx+(L_1/math.sqrt(P_HIGH_1)))) - L_1*math.sqrt(P_LOW_1)
                nextdydx = -L_1**2/(nextx+L_1/math.sqrt(P_HIGH_1))**2
                a=1
        elif aux_L2 == L_2**2:
            if X+deltax<0:
                continue
            elif X+deltax>Xmax2_uni:
                deltax_aux = Xmax2_uni-X
                nextx = deltax-deltax_aux
                if nextx>Xmax1_uni:
                    continue
                else:
                    nexty = (L_1**2 / (nextx+(L_1/math.sqrt(P_HIGH_1)))) - L_1*math.sqrt(P_LOW_1)
                    nextdydx = -L_1**2/(nextx+L_1/math.sqrt(P_HIGH_1))**2
                    a=1
            else:
                nextx = X+deltax
                nexty = (L_2**2 / (nextx+(L_2/math.sqrt(P_HIGH_2)))) - L_2*math.sqrt(P_LOW_2)
                nextdydx = -L_2**2/(nextx+L_2/math.sqrt(P_HIGH_2))**2
                a=1

    if round((nextx+L_1/math.sqrt(P_HIGH_1))*(nexty+L_1*math.sqrt(P_LOW_1)),2) == L_1**2:
        nextXjoin = nextx + Xmax2_uni
        nextYjoin = nexty 
        if b==0:
            Xaux = nextXjoin
            Yaux = nextYjoin
            b=1
    elif round((nextx+L_2/math.sqrt(P_HIGH_2))*(nexty+L_2*math.sqrt(P_LOW_2)),2) == L_2**2:
        nextXjoin = nextx
        nextYjoin = nexty+ Ymax1_uni #K2 / ((nextXjoin) + S1_X0_2*(A-1)) - S1_Y0_2*(A-1) 
         
    xtrade_uni.append([X, nextx])
    ytrade_uni.append([Y, nexty])
    xjointrade_uni.append([Xjoin, nextXjoin])
    yjointrade_uni.append([Yjoin, nextYjoin])
    Yprice_curve_uni.append([dYdX, nextdydx])        
    X, Y = nextx, nexty
    Xjoin, Yjoin = nextXjoin, nextYjoin
    dYdX= nextdydx

# BANCOR
# INITIALIZE SEPARATE BANCOR FUNCTION
def initialize_separate_bonding_curves_bancor(X0,Y0,A):
    Xmax=((A**2)*X0*Y0) / (Y0*(A-1)) - X0*(A-1)
    Ymax=((A**2)*X0*Y0) / (X0*(A-1)) - Y0*(A-1)
    xplot = np.linspace(0, Xmax, 500)
    yplot =((A**2)*X0*Y0) / (xplot+X0*(A-1)) - Y0*(A-1)
    dydxplot = - (A**2 * X0*Y0) / ((xplot + X0 * (A - 1)) ** 2)
    return xplot, yplot, dydxplot, Xmax, Ymax
    
# PLOTS
xplot1_bancor, yplot1_bancor, dydxplot1_bancor, Xmax1_bancor, Ymax1_bancor = initialize_separate_bonding_curves_bancor(S1_X0_1, S1_Y0_1,A)
xplot2_bancor, Yplot2_bancor, dydxplot2_bancor, Xmax2_bancor, Ymax2_bancor = initialize_separate_bonding_curves_bancor(S1_X0_2, S1_Y0_2,A)


def initialize_joint_bancor_curve_1(X0,Y0,A):
    xplot = np.linspace(0, Xmax2_bancor, 500)
    yplot =((A**2)*X0*Y0) / (xplot+X0*(A-1)) - Y0*(A-1) + Ymax1_bancor
    return xplot, yplot

def initialize_joint_bancor_curve_2(X0,Y0,A):
    xplot = np.linspace(Xmax2_bancor, Xmax2_bancor+Xmax1_bancor, 500)
    yplot =((A**2)*X0*Y0) / ((xplot-Xmax2_bancor)+X0*(A-1)) - Y0*(A-1) 
    dydxplot = - (A**2 * X0*Y0) / (((xplot-Xmax2_bancor) + X0 * (A - 1)) ** 2)
    return xplot, yplot, dydxplot

#PLOTS
xplot3, yplot3 = initialize_joint_bancor_curve_1(S1_X0_2, S1_Y0_2, A)
xplot4, yplot4, dydxplot3 = initialize_joint_bancor_curve_2(S1_X0_1, S1_Y0_1, A)
xplot5 = np.linspace(Xmax2_bancor, Xmax2_bancor+Xmax1_bancor, 500)

#START AMMS
X0_reference = S1_X0_2
Y0_reference = S1_Y0_2

X=X0_reference #StartX
Y=Y0_reference #StartY

#TRADE
if round((X+S1_X0_1*(A-1))*(Y+S1_Y0_1*(A-1)),2) == C1:
    Xjoin = X+Xmax2_bancor
    Yjoin = ((A**2)*X*Y) / ((X-Xmax2_bancor)+X*(A-1)) - Y*(A-1) 
    
elif round((X+S1_X0_2*(A-1))*(Y+S1_Y0_2*(A-1)),2) == C2:
    Xjoin = X
    Yjoin = ((A**2)*X*Y) / (X+X*(A-1)) - Y*(A-1) + Ymax1_bancor

#PLOTS
dYdX = -(A**2*X*Y)/(X+X*(A-1))**2

xtrade = []
ytrade = []

xjointrade = []
yjointrade = []

Xprice_curve=[]
Yprice_curve=[]


for _ in range(NUMBER_OF_TRADES):
    a=0
    b=0
    while a==0:
        aux_C1 = round((X+S1_X0_1*(A-1))*(Y+S1_Y0_1*(A-1)),2)
        aux_C2 = round((X+S1_X0_2*(A-1))*(Y+S1_Y0_2*(A-1)),2)
        deltax=norm.rvs(loc=0, scale=300, size=1)[0]
        
        if aux_C1 == C1:
            if X+deltax>Xmax1_bancor:
                continue
            elif X+deltax<0:
                deltax_aux = abs(deltax)-X
                nextx = Xmax2_bancor-deltax_aux
                if nextx<0:
                    continue
                else:
                    nexty = ((S1_X0_2*S1_Y0_2*A**2) / (nextx+S1_X0_2*(A-1)))-S1_Y0_2*(A-1)   
                    nextdydx = -(A**2*S1_X0_2*S1_Y0_2)/(nextx+S1_X0_2*(A-1))**2
                    nextdxdy = -(nextx+S1_X0_2*(A-1))**2/(A**2*S1_X0_2*S1_Y0_2)
                    a=1        
            else:
                nextx = X+deltax
                nexty = ((S1_X0_1*S1_Y0_1*A**2) / (nextx+S1_X0_1*(A-1)))-S1_Y0_1*(A-1)
                nextdydx = -(A**2*S1_X0_1*S1_Y0_1)/(nextx+S1_X0_1*(A-1))**2
                nextdxdy = -(nextx+S1_X0_1*(A-1))**2/(A**2*S1_X0_1*S1_Y0_1)
                a=1
        elif aux_C2 == C2:
            if X+deltax<0:
                continue
            elif X+deltax>Xmax2_bancor:
                deltax_aux = Xmax2_bancor-X
                nextx = deltax-deltax_aux
                if nextx>Xmax1_bancor:
                    continue
                else:
                    nexty = ((S1_X0_1*S1_Y0_1*A**2) / (nextx+S1_X0_1*(A-1)))-S1_Y0_1*(A-1) 
                    nextdydx = -(A**2*S1_X0_1*S1_Y0_1)/(nextx+S1_X0_1*(A-1))**2
                    nextdxdy = -(nextx+S1_X0_1*(A-1))**2/(A**2*S1_X0_1*S1_Y0_1)
                    a=1
            else:
                nextx = X+deltax
                nexty = ((S1_X0_2*S1_Y0_2*A**2) / (nextx+S1_X0_2*(A-1)))-S1_Y0_2*(A-1) 
                nextdydx = -(A**2*S1_X0_2*S1_Y0_2)/((nextx+S1_X0_2*(A-1))**2)
                nextdxdy = -(nextx+S1_X0_2*(A-1))**2/(A**2*S1_X0_2*S1_Y0_2)
                a=1

    if round((nextx+S1_X0_1*(A-1))*(nexty+S1_Y0_1*(A-1)),2) == C1:
        nextXjoin = nextx + Xmax2_bancor
        nextYjoin = nexty 
        if b==0:
            Xaux = nextXjoin
            Yaux = nextYjoin
            b=1
    elif round((nextx+S1_X0_2*(A-1))*(nexty+S1_Y0_2*(A-1)),2) == C2:
        nextXjoin = nextx
        nextYjoin = nexty+ Ymax1_bancor#K2 / ((nextXjoin) + S1_X0_2*(A-1)) - S1_Y0_2*(A-1) 
         
    xtrade.append([X, nextx])
    ytrade.append([Y, nexty])
    xjointrade.append([Xjoin, nextXjoin])
    yjointrade.append([Yjoin, nextYjoin])
    Yprice_curve.append([dYdX, nextdydx])        
    X, Y = nextx, nexty
    Xjoin, Yjoin = nextXjoin, nextYjoin
    dYdX= nextdydx
    


# Creating subplots for each trade
for i in range(NUMBER_OF_TRADES):
    # Create a figure
    fig = plt.figure(figsize=(16, 10.5), facecolor='#eeeef2')

    # Define width and height ratios for each column and row respectively
    width_ratios = [1, 2.25, 2.25, 2.25, 2.25]
    height_ratios = [0.5, 4.75, 4.75]

    # Create a GridSpec with 2x5 layout and specify width and height ratios
    gs = gridspec.GridSpec(3, 5, figure=fig, width_ratios=width_ratios, height_ratios=height_ratios)

    # Add subplots to the GridSpec and store them in a 2D list
    axes = []
    for row in range(3):
        for col in range(5):
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)
    
    axes[0].axis('off')
    axes[0].text(
        0.5,
        0, 
        f'Trade Nº {i + 1}', 
        ha='center', va='center', fontsize=15, fontstyle='italic',fontweight='bold'
    )
    
    axes[1].axis('off')
    axes[1].text(
        0.5,
        0.5, 
        "Separate Bonding Curves", 
        ha='center', va='center', fontsize=12, fontstyle='italic',fontweight='bold'
    )
    axes[1].text(
        0.5,
        -0.6, 
        f'Bancor C1: (X+{S1_X0_2*(A-1)}) * (Y+{S1_Y0_2*(A-1)}) = {int(round(A**2*S1_X0_2*S1_Y0_2,0))}\nBancor C2: (X+{S1_X0_1*(A-1)}) * (Y+{S1_Y0_1*(A-1)}) = {int(round(A**2*S1_X0_1*S1_Y0_1,0))}\nUniswap C1: (X+{round(L_1/math.sqrt(P_HIGH_1),1)})*(Y+{round(L_1*math.sqrt(P_LOW_1),1)})={L_1**2}\nUniswap C2: (X+{round(L_2/math.sqrt(P_HIGH_2),1)})*(Y+{round(L_2*math.sqrt(P_LOW_2),1)})={L_2**2}',
        ha='center', va='center', fontsize=8, fontstyle='italic',fontweight='light'
    )
    
    axes[2].axis('off')
    axes[2].text(
        0.5,
        0.5, 
        "Joint Bonding Curves", 
        ha='center', va='center', fontsize=12, fontstyle='italic',fontweight='bold'
    )
    axes[2].text(
        0.5,
        -0.6, 
        f'Bancor C1: (X+{S1_X0_2*(A-1)}) * (Y+{(S1_Y0_2+Ymax1_bancor)*(A-1)}) = {int(round(A**2*S1_X0_2*(S1_Y0_2+Ymax1_bancor),0))}\nBancor C2: (X+{(S1_X0_1+Xmax2_bancor)*(A-1)}) * (Y+{S1_Y0_1*(A-1)}) = {int(round(A**2*(S1_X0_1+Xmax2_bancor)*S1_Y0_1,0))}\nUniswap C1: (X+{round(L_1/math.sqrt(P_HIGH_1),1)})*(Y+{round(L_1*math.sqrt(P_LOW_1),1)})={L_1**2}\nUniswap C2:',
        ha='center', va='center', fontsize=8, fontstyle='italic',fontweight='light'
    )
    
    axes[3].axis('off')
    axes[3].text(
        0.5,
        0.5, 
        "Separate dY/dX Price Curves", 
        ha='center', va='center', fontsize=12, fontstyle='italic',fontweight='bold'
    )
    axes[3].text(
        0.5,
        -0.4, 
        f"C1: {-A**2*S1_X0_2*S1_Y0_2} / (X+{S1_X0_2*(A - 1)})^2\nC2: {-A**2*S1_X0_1*S1_Y0_1} / (X+{S1_X0_1*(A - 1)})^2\n", 
        ha='center', va='center', fontsize=8, fontstyle='italic',fontweight='light'
    )
    #- (A**2 * X0*Y0) / ((xplot + X0 * (A - 1)) ** 2)
    
    axes[4].axis('off')
    axes[4].text(
        0.5,
        0.5, 
        "Joint dY/dX Price Curves", 
        ha='center', va='center', fontsize=12, fontstyle='italic',fontweight='bold'
    )
    axes[4].text(
        0.5,
        -0.4, 
        f"C1: {-A**2*S1_X0_2*S1_Y0_2} / (X+{S1_X0_2*(A - 1)})^2\nC2: {-A**2*(S1_X0_1+Xmax2_bancor)*S1_Y0_1} / (X+{(S1_X0_1+Xmax2_bancor)*(A - 1)})^2\n", 
        ha='center', va='center', fontsize=8, fontstyle='italic',fontweight='light' 
    )

    axes[5].axis('off')
    axes[5].text(
        0.5,
        0.5, 
        "Bancor v2\nReal Curve", 
        ha='center', va='center', fontsize=12, fontstyle='italic',fontweight='bold'
    )
    axes[5].text(
        0.5,
        0.35, 
        f'A={A}\nX0_1={S1_X0_2} | Y0_1={S1_Y0_2}\nX0_2={S1_X0_1} | Y0_2={S1_Y0_1}', 
        ha='center', va='center', fontsize=9, fontstyle='italic',fontweight='light'
    )
    
    #ax6
    axes[6].plot(xplot1_bancor, yplot1_bancor, color='blue')
    axes[6].plot(xplot2_bancor, Yplot2_bancor, color='red')
    axes[6].scatter(
        [xtrade[i][0], xtrade[i][1], 0, Xmax1_bancor],
        [ytrade[i][0], ytrade[i][1], Ymax2_bancor, 0], 
        color=['red', 'blue', 'orange', 'orange'], 
        edgecolors=['black', 'black', 'red', 'blue'],
        s=80
    ) 
    
    axes[6].set_xlabel("Token X Balance")
    axes[6].set_ylabel("Token Y Balance")
    axes[6].grid(True) 
    
    ax1_elements = [
        mlines.Line2D([], [], color='red', markersize=10, label='Bonding Curve 1'),
        mlines.Line2D([], [], color='blue', markersize=10, label='Bonding Curve 2'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='red', markersize=10, label=f'Before trade ({round(xtrade[i][0],2)} , {round(ytrade[i][0],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='blue', markersize=10, label=f'After trade ({round(xtrade[i][1],2)} , {round(ytrade[i][1],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='orange',markeredgecolor='red', markersize=10, label=f'Price Bound 1 (0 , {round(Ymax2_bancor,2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='orange',markeredgecolor='blue', markersize=10, label=f'Price Bound 2 ({round(Xmax1_bancor,2)} , 0)'),
        mlines.Line2D([], [], color='none', markersize=0, label=f'Effective Rate = {round((yjointrade[i][1]-yjointrade[i][0])/(xjointrade[i][1]-xjointrade[i][0]),2)}'),
    ]

    axes[6].legend(handles=ax1_elements, loc='best', shadow=True, fancybox=True, title='Legend', title_fontsize='7', fontsize='7', framealpha=1)
    
    # Adding arrows    
    if xtrade[i][1]-xtrade[i][0]<0:
        arrow1 = FancyArrowPatch((xtrade[i][0], ytrade[i][0]), (xtrade[i][0], ytrade[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow2 = FancyArrowPatch((xtrade[i][0], ytrade[i][1]), (xtrade[i][1], ytrade[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow3 = FancyArrowPatch((xtrade[i][0], ytrade[i][0]), (xtrade[i][1], ytrade[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    elif xtrade[i][1]-xtrade[i][0]>0:
        arrow1 = FancyArrowPatch((xtrade[i][0], ytrade[i][0]), (xtrade[i][1], ytrade[i][0]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow2 = FancyArrowPatch((xtrade[i][1], ytrade[i][0]), (xtrade[i][1], ytrade[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow3 = FancyArrowPatch((xtrade[i][0], ytrade[i][0]), (xtrade[i][1], ytrade[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    axes[6].add_patch(arrow1)
    axes[6].add_patch(arrow2)
    axes[6].add_patch(arrow3)
    
    # axes[7].axis('off')
    axes[7].plot(xplot4, yplot4, color='blue')
    axes[7].plot(xplot3, yplot3, color='red')
    axes[7].scatter(
        [xjointrade[i][0], xjointrade[i][1], 0, Xmax1_bancor+Xmax2_bancor, Xmax2_bancor],
        [yjointrade[i][0], yjointrade[i][1], Ymax2_bancor+Ymax1_bancor, 0, Ymax1_bancor], 
        color=['red', 'blue', 'orange','orange', 'green'], 
        edgecolors=['black', 'black', 'red', 'blue', 'black'],
        s=80
    )
    
    axes[7].set_xlabel("Token X Balance")
    axes[7].set_ylabel("Token Y Balance")
    axes[7].grid(True) 
    
    ax2_elements = [
        mlines.Line2D([], [], color='red', markersize=10, label='Bonding Curve 1'),
        mlines.Line2D([], [], color='blue', markersize=10, label='Bonding Curve 2'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='red', markersize=10, label=f'Before Trade ({round(xjointrade[i][0],2)} , {round(yjointrade[i][0],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='blue', markersize=10, label=f'After Trade ({round(xjointrade[i][1],2)} , {round(yjointrade[i][1],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='orange',markeredgecolor='red', markersize=10, label=f'Price Bound 1 (0 , {round(Ymax2_bancor+Ymax1_bancor,2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='orange',markeredgecolor='blue', markersize=10, label=f'Price Bound 2 ({round(Xmax1_bancor+Xmax2_bancor,2)} , 0)'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='green',markeredgecolor='black', markersize=10, label=f'P_Low1 = P_High2 ({Xmax2_bancor} , {Ymax1_bancor})'),
        mlines.Line2D([], [], color='none', markersize=0, label=f'Effective Rate = {round((yjointrade[i][1]-yjointrade[i][0])/(xjointrade[i][1]-xjointrade[i][0]),2)}'),
    ]
    axes[7].legend(handles=ax2_elements, loc='best', shadow=True, fancybox=True, title='Legend', title_fontsize='7', fontsize='7', framealpha=1)
    
    # Adding arrows
    if xjointrade[i][1]-xjointrade[i][0]<0:
        arrow4 = FancyArrowPatch((xjointrade[i][0], yjointrade[i][0]), (xjointrade[i][0], yjointrade[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow5 = FancyArrowPatch((xjointrade[i][0], yjointrade[i][1]), (xjointrade[i][1], yjointrade[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow6 = FancyArrowPatch((xjointrade[i][0], yjointrade[i][0]), (xjointrade[i][1], yjointrade[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    elif xjointrade[i][1]-xjointrade[i][0]>0:
        arrow4 = FancyArrowPatch((xjointrade[i][0], yjointrade[i][0]), (xjointrade[i][1], yjointrade[i][0]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow5 = FancyArrowPatch((xjointrade[i][1], yjointrade[i][0]), (xjointrade[i][1], yjointrade[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow6 = FancyArrowPatch((xjointrade[i][0], yjointrade[i][0]), (xjointrade[i][1], yjointrade[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    axes[7].add_patch(arrow4)
    axes[7].add_patch(arrow5)
    axes[7].add_patch(arrow6)
    
    
    # axes[8].axis('off')
    axes[8].plot(xplot1_bancor, dydxplot1_bancor, color='blue')
    axes[8].plot(xplot2_bancor, dydxplot2_bancor, color='red')
    
    [ytrade[i][0], ytrade[i][1], Ymax2_bancor, 0], 
    
    axes[8].scatter(
        [xtrade[i][0], xtrade[i][1], 0, Xmax1_bancor],
        [Yprice_curve[i][0], Yprice_curve[i][1], dydxplot2_bancor[0], dydxplot1_bancor[-1]], 
        color=['red', 'blue', 'orange', 'orange'], 
        edgecolors=['black', 'black', 'red', 'blue'],
        s=80
    )
    
    ax3_elements = [
        mlines.Line2D([], [], color='red', markersize=10, label='dY/dX Curve 1'),
        mlines.Line2D([], [], color='blue', markersize=10, label='dY/dX Curve 2'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='red', markeredgecolor='black', markersize=10, label=f'Before Trade ({round(xtrade[i][0],2)} , {round(Yprice_curve[i][0],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='blue', markeredgecolor='black',markersize=10, label=f'After Trade ({round(xtrade[i][1],2)} , {round(Yprice_curve[i][1],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='orange',markeredgecolor='red', markersize=10, label=f'Price Bound 1 (0 , {round(dydxplot2_bancor[0],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='orange',markeredgecolor='blue', markersize=10, label=f'Price Bound 2 ({round(Xmax1_bancor,2)} , {round(dydxplot1_bancor[-1],2)})'),
        # mlines.Line2D([], [], color='none', label=f'|ΔX|={abs(xtrade[i][1]-xtrade[i][0]):.2f}'),
    ]
    axes[8].legend(handles=ax3_elements, loc='best', shadow=True, fancybox=True, title='Legend', title_fontsize='7', fontsize='7', framealpha=1)

    axes[8].set_xlabel("Token X Balance")
    axes[8].set_ylabel("dY / dX")
    axes[8].grid(True) 
    
    # Integral Area
    # sectionx = np.linspace(xjointrade[i][0], xjointrade[i][1], 500)
    if round((xtrade[i][0]+S1_X0_1*(A-1))*(ytrade[i][0]+S1_Y0_1*(A-1)),2)==round((xtrade[i][1]+S1_X0_1*(A-1))*(ytrade[i][1]+S1_Y0_1*(A-1)),2)==C1:
        sectionx = np.linspace(xtrade[i][0], xtrade[i][1], 500)
        section_dydx = -((A**2)*S1_X0_1*S1_Y0_1) / ((sectionx+S1_X0_1 *(A-1))**2)  
        axes[8].fill_between(sectionx, section_dydx, alpha=0.5, color ='#afaffd') 
        
    elif round((xtrade[i][0]+S1_X0_2*(A-1))*(ytrade[i][0]+S1_Y0_2*(A-1)),2)==round((xtrade[i][1]+S1_X0_2*(A-1))*(ytrade[i][1]+S1_Y0_2*(A-1)),2)==C2: #CHECK
        sectionx = np.linspace(xjointrade[i][0], xjointrade[i][1], 500) 
        section_dydx = -((A**2)*S1_X0_2*S1_Y0_2) / ((sectionx+S1_X0_2*(A-1))**2) 
        axes[8].fill_between(sectionx, section_dydx, alpha=0.5, color ='#fdbcbc') 
        
    elif round((xtrade[i][0]+S1_X0_2*(A-1))*(ytrade[i][0]+S1_Y0_2*(A-1)),2)==C2 and round((xtrade[i][1]+S1_X0_1*(A-1))*(ytrade[i][1]+S1_Y0_1*(A-1)),2)==C1:
        sectionx = np.linspace(xtrade[i][0], Xmax2_bancor, 500)
        section_dydx = -((A**2)*S1_X0_2*S1_Y0_2) / ((sectionx+S1_X0_2*(A-1))**2)
        axes[8].fill_between(sectionx, section_dydx, alpha=0.5, color ='#fdbcbc')
        
        sectionx = np.linspace(0, xtrade[i][1], 500)
        section_dydx = -((A**2)*S1_X0_1*S1_Y0_1) / ((sectionx+S1_X0_1*(A-1))**2)  
        axes[8].fill_between(sectionx, section_dydx, alpha=0.5, color ='#afaffd') 
    elif round((xtrade[i][0]+S1_X0_1*(A-1))*(ytrade[i][0]+S1_Y0_1*(A-1)),2)==C1 and round((xtrade[i][1]+S1_X0_2*(A-1))*(ytrade[i][1]+S1_Y0_2*(A-1)),2)==C2:
        sectionx = np.linspace(xtrade[i][0], 0, 500)
        section_dydx = -((A**2)*S1_X0_1*S1_Y0_1) / ((sectionx+S1_X0_1*(A-1))**2)
        axes[8].fill_between(sectionx, section_dydx, alpha=0.5, color ='#afaffd')
        
        sectionx = np.linspace( Xmax2_bancor, xtrade[i][1], 500)
        section_dydx = -((A**2)*S1_X0_2*S1_Y0_2) / ((sectionx+S1_X0_2*(A-1))**2)  
        axes[8].fill_between(sectionx, section_dydx, alpha=0.5, color ='#fdbcbc') 
    
    # Adding arrows
    if xtrade[i][1]-xtrade[i][0]<0:
        arrow7 = FancyArrowPatch((xtrade[i][0], Yprice_curve[i][0]), (xtrade[i][0], Yprice_curve[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow8 = FancyArrowPatch((xtrade[i][0], Yprice_curve[i][1]), (xtrade[i][1], Yprice_curve[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow9 = FancyArrowPatch((xtrade[i][0], Yprice_curve[i][0]), (xtrade[i][1], Yprice_curve[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    elif xtrade[i][1]-xtrade[i][0]>0:
        arrow7 = FancyArrowPatch((xtrade[i][0], Yprice_curve[i][0]), (xtrade[i][1], Yprice_curve[i][0]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow8 = FancyArrowPatch((xtrade[i][1], Yprice_curve[i][0]), (xtrade[i][1], Yprice_curve[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow9 = FancyArrowPatch((xtrade[i][0], Yprice_curve[i][0]), (xtrade[i][1], Yprice_curve[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    
    axes[8].add_patch(arrow7)
    axes[8].add_patch(arrow8)
    axes[8].add_patch(arrow9)
    axes[8].legend(handles=ax3_elements, loc='best', shadow=True, fancybox=True, title='Legend', title_fontsize='7', fontsize='7', framealpha=1)
    
    # axes[9].axis('off')
    axes[9].plot(xplot5, dydxplot1_bancor, color='blue')
    axes[9].plot(xplot2_bancor, dydxplot2_bancor, color='red')
    axes[9].scatter(
        [xjointrade[i][0], xjointrade[i][1], 0, Xmax1_bancor+Xmax2_bancor, Xmax2_bancor],
        [Yprice_curve[i][0], Yprice_curve[i][1], dydxplot2_bancor[0], dydxplot1_bancor[-1], dydxplot2_bancor[-1]], 
        color=['red', 'blue', 'orange', 'orange', 'green'], 
        edgecolors=['black', 'black', 'red', 'blue', 'black'],
        s=80
    )
    
    ax4_elements = [
        mlines.Line2D([], [], color='red', markersize=10, label='dY/dX Curve 1'),
        mlines.Line2D([], [], color='blue', markersize=10, label='dY/dX Curve 2'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='red', markeredgecolor='black', markersize=10, label=f'Before Trade ({round(xjointrade[i][0],2)} , {round(Yprice_curve[i][0],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='blue', markeredgecolor='black',markersize=10, label=f'After Trade ({round(xjointrade[i][1],2)} , {round(Yprice_curve[i][1],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='orange',markeredgecolor='red', markersize=10, label=f'Price Bound 1 (0 , {round(dydxplot2_bancor[0],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='orange',markeredgecolor='blue', markersize=10, label=f'Price Bound 2 ({round(Xmax1_bancor+Xmax2_bancor,2)} , {round(dydxplot1_bancor[-1],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='green', markeredgecolor='black',markersize=10, label=f'P_Low1 = P_High2 ({round(Xmax2_bancor,2)} , {round(dydxplot2_bancor[-1],2)})')
    ]
    axes[9].legend(handles=ax4_elements, loc='best', shadow=True, fancybox=True, title='Legend', title_fontsize='7', fontsize='7', framealpha=1)

    axes[9].set_xlabel("Token X Balance")
    axes[9].set_ylabel("dY / dX")
    axes[9].grid(True) 
    
    # Integral Area
    # sectionx = np.linspace(xjointrade[i][0], xjointrade[i][1], 500)
    if Xmax2_bancor<xjointrade[i][0] and Xmax2_bancor<xjointrade[i][1]:
        sectionx = np.linspace(xjointrade[i][0], xjointrade[i][1], 500)
        section_dydx = -((A**2)*S1_X0_1*S1_Y0_1) / (((sectionx-Xmax2_bancor)+S1_X0_1 *(A-1))**2)  
        axes[9].fill_between(sectionx, section_dydx, alpha=0.5, color ='#afaffd') 
    elif Xmax2_bancor>xjointrade[i][0] and Xmax2_bancor>xjointrade[i][1]:
        sectionx = np.linspace(xjointrade[i][0], xjointrade[i][1], 500)
        section_dydx = -((A**2)*S1_X0_2*S1_Y0_2) / ((sectionx+S1_X0_2*(A-1))**2) 
        axes[9].fill_between(sectionx, section_dydx, alpha=0.5, color ='#fdbcbc') 
    elif (Xmax2_bancor>xjointrade[i][0] and Xmax2_bancor<xjointrade[i][1]):
        sectionx = np.linspace(xjointrade[i][0], Xmax2_bancor, 500)
        section_dydx = -((A**2)*S1_X0_2*S1_Y0_2) / ((sectionx+S1_X0_2*(A-1))**2)
        axes[9].fill_between(sectionx, section_dydx, alpha=0.5, color ='#fdbcbc')
        Xaux= xjointrade[i][1]-Xmax2_bancor
        sectionx = np.linspace(Xmax2_bancor, xjointrade[i][1], 500)
        section_dydx = -((A**2)*S1_X0_1*S1_Y0_1) / (((sectionx-Xmax2_bancor)+S1_X0_1 *(A-1))**2)  
        axes[9].fill_between(sectionx, section_dydx, alpha=0.5, color ='#afaffd') 
    elif (Xmax2_bancor<xjointrade[i][0] and Xmax2_bancor>xjointrade[i][1]):
        sectionx = np.linspace(xjointrade[i][1], Xmax2_bancor, 500)
        section_dydx = -((A**2)*S1_X0_2*S1_Y0_2) / ((sectionx+S1_X0_2*(A-1))**2)
        axes[9].fill_between(sectionx, section_dydx, alpha=0.5, color ='#fdbcbc')
        sectionx = np.linspace(Xmax2_bancor, xjointrade[i][0], 500)
        section_dydx = -((A**2)*S1_X0_1*S1_Y0_1) / (((sectionx-Xmax2_bancor)+S1_X0_1 *(A-1))**2)  
        axes[9].fill_between(sectionx, section_dydx, alpha=0.5, color ='#afaffd') 
       
    # Adding arrows
    if xtrade[i][1]-xtrade[i][0]<0:  #round(Xmax1_bancor+Xmax2_bancor,2)
        arrow10 = FancyArrowPatch((xjointrade[i][0], Yprice_curve[i][0]), (xjointrade[i][0], Yprice_curve[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow11 = FancyArrowPatch((xjointrade[i][0], Yprice_curve[i][1]), (xjointrade[i][1], Yprice_curve[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow12 = FancyArrowPatch((xjointrade[i][0], Yprice_curve[i][0]), (xjointrade[i][1], Yprice_curve[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    elif xtrade[i][1]-xtrade[i][0]>0:
        arrow10 = FancyArrowPatch((xjointrade[i][0], Yprice_curve[i][0]), (xjointrade[i][1], Yprice_curve[i][0]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow11 = FancyArrowPatch((xjointrade[i][1], Yprice_curve[i][0]), (xjointrade[i][1], Yprice_curve[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow12 = FancyArrowPatch((xjointrade[i][0], Yprice_curve[i][0]), (xjointrade[i][1], Yprice_curve[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    
    axes[9].add_patch(arrow10)
    axes[9].add_patch(arrow11)
    axes[9].add_patch(arrow12)
    
    # sddsds
    axes[10].axis('off')
    axes[10].text(
        0.5,
        0.5, 
        "Uniswap v3\nReal Curve", 
        ha='center', va='center', fontsize=12, fontstyle='italic',fontweight='bold'
    )
    axes[10].text(
        0.5,
        0.35, 
        f'L1={L_1} | L2={L_2}\nP_LOW_1={P_LOW_1} | P_LOW_2={P_LOW_2}\nP_HIGH_1={P_HIGH_1} | P_HIGH_2={P_HIGH_2}',  
        ha='center', va='center', fontsize=8, fontstyle='italic',fontweight='light'
    )
    
    #ax6
    axes[11].plot(xplot1_uni, yplot1_uni, color='blue')
    axes[11].plot(xplot2_uni, yplot2_uni, color='red')
    axes[11].scatter(
        [0, Xmax1_uni, xtrade_uni[i][0], xtrade_uni[i][1]],
        [Ymax2_uni , 0, ytrade_uni[i][0], ytrade_uni[i][1]], 
        color=['orange', 'orange','red','blue'], 
        edgecolors=['red', 'blue','black','black'],
        s=80
    ) 
    
    axes[11].set_xlabel("Token X Balance")
    axes[11].set_ylabel("Token Y Balance")
    axes[11].grid(True) 
    
    ax11_elements = [
        mlines.Line2D([], [], color='red', markersize=10, label='Bonding Curve 1'),
        mlines.Line2D([], [], color='blue', markersize=10, label='Bonding Curve 2'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='red', markersize=10, label=f'Before trade ({round(xtrade_uni[i][0],2)} , {round(ytrade_uni[i][0],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='blue', markersize=10, label=f'After trade ({round(xtrade_uni[i][1],2)} , {round(ytrade_uni[i][1],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='orange',markeredgecolor='red', markersize=10, label=f'Price Bound 1 (0 , {round(yplot2_uni[0],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='orange',markeredgecolor='blue', markersize=10, label=f'Price Bound 2 ({round(xplot1_uni[-1],2)} , 0)'),
        # mlines.Line2D([], [], color='none', markersize=0, label=f'Effective Rate = {round((yjointrade[i][1]-yjointrade[i][0])/(xjointrade[i][1]-xjointrade[i][0]),2)}'),
    ]
    axes[11].legend(handles=ax11_elements, loc='best', shadow=True, fancybox=True, title='Legend', title_fontsize='7', fontsize='7', framealpha=1)
    
    # Adding arrows    
    if xtrade_uni[i][1]-xtrade_uni[i][0]<0:
        arrow13 = FancyArrowPatch((xtrade_uni[i][0], ytrade_uni[i][0]), (xtrade_uni[i][0], ytrade_uni[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow14 = FancyArrowPatch((xtrade_uni[i][0], ytrade_uni[i][1]), (xtrade_uni[i][1], ytrade_uni[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow15 = FancyArrowPatch((xtrade_uni[i][0], ytrade_uni[i][0]), (xtrade_uni[i][1], ytrade_uni[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    elif xtrade_uni[i][1]-xtrade_uni[i][0]>0:
        arrow13 = FancyArrowPatch((xtrade_uni[i][0], ytrade_uni[i][0]), (xtrade_uni[i][1], ytrade_uni[i][0]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow14 = FancyArrowPatch((xtrade_uni[i][1], ytrade_uni[i][0]), (xtrade_uni[i][1], ytrade_uni[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow15 = FancyArrowPatch((xtrade_uni[i][0], ytrade_uni[i][0]), (xtrade_uni[i][1], ytrade_uni[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    axes[11].add_patch(arrow13)
    axes[11].add_patch(arrow14)
    axes[11].add_patch(arrow15)
    
    axes[12].plot(xplot3_uni, yplot3_uni, color='blue')
    axes[12].plot(xplot4_uni, yplot4_uni, color='red')
    axes[12].scatter(
        [0, Xmax1_uni+Xmax2_uni, xplot4_uni[-1], xjointrade_uni[i][0], xjointrade_uni[i][1]],
        [Ymax2_uni+Ymax1_uni , 0, yplot4_uni[-1], yjointrade_uni[i][0], yjointrade_uni[i][1]], 
        color=['orange', 'orange','green','red','blue'], 
        edgecolors=['red', 'blue', 'black','black','black'],
        s=80
    ) 
    
    axes[12].set_xlabel("Token X Balance")
    axes[12].set_ylabel("Token Y Balance")
    axes[12].grid(True) 
    
    ax12_elements = [
        mlines.Line2D([], [], color='red', markersize=10, label='Bonding Curve 1'),
        mlines.Line2D([], [], color='blue', markersize=10, label='Bonding Curve 2'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='red', markersize=10, label=f'Before Trade ({round(xjointrade_uni[i][0],2)} , {round(yjointrade_uni[i][0],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='blue', markersize=10, label=f'After Trade ({round(xjointrade_uni[i][1],2)} , {round(yjointrade_uni[i][1],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='orange',markeredgecolor='red', markersize=10, label=f'Price Bound 1 (0 , {round(yplot4_uni[0],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='orange',markeredgecolor='blue', markersize=10, label=f'Price Bound 2 ({round(xplot3_uni[-1],2)} , 0)'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='green',markeredgecolor='black', markersize=10, label=f'P_Low1 = P_High2 ({round(xplot3_uni[0],2)} , {round(yplot3_uni[0],2)})'),
        # mlines.Line2D([], [], color='none', markersize=0, label=f'Effective Rate = {round((yjointrade[i][1]-yjointrade[i][0])/(xjointrade[i][1]-xjointrade[i][0]),2)}'),
    ]
    axes[12].legend(handles=ax12_elements, loc='best', shadow=True, fancybox=True, title='Legend', title_fontsize='7', fontsize='7', framealpha=1)
    
    # Adding arrows
    if xjointrade_uni[i][1]-xjointrade_uni[i][0]<0:
        arrow16 = FancyArrowPatch((xjointrade_uni[i][0], yjointrade_uni[i][0]), (xjointrade_uni[i][0], yjointrade_uni[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow17 = FancyArrowPatch((xjointrade_uni[i][0], yjointrade_uni[i][1]), (xjointrade_uni[i][1], yjointrade_uni[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow18 = FancyArrowPatch((xjointrade_uni[i][0], yjointrade_uni[i][0]), (xjointrade_uni[i][1], yjointrade_uni[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    elif xjointrade_uni[i][1]-xjointrade_uni[i][0]>0:
        arrow16 = FancyArrowPatch((xjointrade_uni[i][0], yjointrade_uni[i][0]), (xjointrade_uni[i][1], yjointrade_uni[i][0]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow17 = FancyArrowPatch((xjointrade_uni[i][1], yjointrade_uni[i][0]), (xjointrade_uni[i][1], yjointrade_uni[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow18 = FancyArrowPatch((xjointrade_uni[i][0], yjointrade_uni[i][0]), (xjointrade_uni[i][1], yjointrade_uni[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    axes[12].add_patch(arrow16)
    axes[12].add_patch(arrow17)
    axes[12].add_patch(arrow18)
    
    
    axes[13].plot(xplot1_uni, dydxplot1_uni, color='blue')
    axes[13].plot(xplot2_uni, dydxplot2_uni, color='red')
    axes[13].scatter(
        [0, xplot1_uni[-1], xtrade_uni[i][0], xtrade_uni[i][1]],
        [dydxplot2_uni[0], dydxplot1_uni[-1], Yprice_curve_uni[i][0], Yprice_curve_uni[i][1]], 
        color=['orange', 'orange', 'red', 'blue'], 
        edgecolors=['red', 'blue', 'black', 'black'],
        s=80
    ) 
    axes[13].set_xlabel("Token X Balance")
    axes[13].set_ylabel("dY / dX")
    axes[13].grid(True) 
    
    ax13_elements = [
        mlines.Line2D([], [], color='red', markersize=10, label='dY/dX Curve 1'),
        mlines.Line2D([], [], color='blue', markersize=10, label='dY/dX Curve 2'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='red', markeredgecolor='black', markersize=10, label=f'Before Trade ({round(xtrade_uni[i][0],2)} , {round(Yprice_curve_uni[i][0],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='blue', markeredgecolor='black',markersize=10, label=f'After Trade ({round(xtrade_uni[i][1],2)} , {round(Yprice_curve_uni[i][1],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='orange',markeredgecolor='red', markersize=10, label=f'Price Bound 1 (0 , {round(dydxplot2_uni[0],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='orange',markeredgecolor='blue', markersize=10, label=f'Price Bound 2 ({round(xplot1_uni[-1],2)} , {round(dydxplot1_uni[-1],2)})'),
        # mlines.Line2D([], [], color='none', markersize=0, label=f'Effective Rate = {round((yjointrade[i][1]-yjointrade[i][0])/(xjointrade[i][1]-xjointrade[i][0]),2)}'),
    ]
    axes[13].legend(handles=ax13_elements, loc='best', shadow=True, fancybox=True, title='Legend', title_fontsize='7', fontsize='7', framealpha=1)

    # Adding arrows
    if xtrade_uni[i][1]-xtrade_uni[i][0]<0:
        arrow19 = FancyArrowPatch((xtrade_uni[i][0], Yprice_curve_uni[i][0]), (xtrade_uni[i][0], Yprice_curve_uni[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow20 = FancyArrowPatch((xtrade_uni[i][0], Yprice_curve_uni[i][1]), (xtrade_uni[i][1], Yprice_curve_uni[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow21 = FancyArrowPatch((xtrade_uni[i][0], Yprice_curve_uni[i][0]), (xtrade_uni[i][1], Yprice_curve_uni[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    elif xtrade_uni[i][1]-xtrade_uni[i][0]>0:
        arrow19 = FancyArrowPatch((xtrade_uni[i][0], Yprice_curve_uni[i][0]), (xtrade_uni[i][1], Yprice_curve_uni[i][0]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow20 = FancyArrowPatch((xtrade_uni[i][1], Yprice_curve_uni[i][0]), (xtrade_uni[i][1], Yprice_curve_uni[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow21 = FancyArrowPatch((xtrade_uni[i][0], Yprice_curve_uni[i][0]), (xtrade_uni[i][1], Yprice_curve_uni[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    
    axes[13].add_patch(arrow19)
    axes[13].add_patch(arrow20)
    axes[13].add_patch(arrow21)
    
    # Integral Area
    # sectionx = np.linspace(xjointrade[i][0], xjointrade[i][1], 500)
    if round(((xtrade_uni[i][0]+L_1/math.sqrt(P_HIGH_1))*(ytrade_uni[i][0]+L_1*math.sqrt(P_LOW_1))),2)==round(((xtrade_uni[i][1]+L_1/math.sqrt(P_HIGH_1))*(ytrade_uni[i][1]+L_1*math.sqrt(P_LOW_1))),2)==L_1**2:
        sectionx = np.linspace(xtrade_uni[i][0], xtrade_uni[i][1], 500)
        section_dydx = -L_1**2/(sectionx+L_1/math.sqrt(P_HIGH_1))**2
        axes[13].fill_between(sectionx, section_dydx, alpha=0.5, color ='#afaffd') 
    elif round(((xtrade_uni[i][0]+L_2/math.sqrt(P_HIGH_2))*(ytrade_uni[i][0]+L_2*math.sqrt(P_LOW_2))),2)==round(((xtrade_uni[i][1]+L_2/math.sqrt(P_HIGH_2))*(ytrade_uni[i][1]+L_2*math.sqrt(P_LOW_2))),2)==L_2**2:
        sectionx = np.linspace(xjointrade_uni[i][0], xjointrade_uni[i][1], 500)                    
        section_dydx =-L_2**2/(sectionx+L_2/math.sqrt(P_HIGH_2))**2
        axes[13].fill_between(sectionx, section_dydx, alpha=0.5, color ='#fdbcbc') 
    elif round(((xtrade_uni[i][0]+L_2/math.sqrt(P_HIGH_2))*(ytrade_uni[i][0]+L_2*math.sqrt(P_LOW_2))),2)==L_2**2 and round(((xtrade_uni[i][1]+L_1/math.sqrt(P_HIGH_1))*(ytrade_uni[i][1]+L_1*math.sqrt(P_LOW_1))),2)==L_1**2:
        sectionx = np.linspace(xtrade_uni[i][0], Xmax2_uni, 500)
        section_dydx = -L_2**2/(sectionx+L_2/math.sqrt(P_HIGH_2))**2
        axes[13].fill_between(sectionx, section_dydx, alpha=0.5, color ='#fdbcbc')

        sectionx = np.linspace(0, xtrade_uni[i][1], 500)
        section_dydx = -L_1**2/(sectionx+L_1/math.sqrt(P_HIGH_1))**2
        axes[13].fill_between(sectionx, section_dydx, alpha=0.5, color ='#afaffd') 
    elif round(((xtrade_uni[i][0]+L_1/math.sqrt(P_HIGH_1))*(ytrade_uni[i][0]+L_1*math.sqrt(P_LOW_1))),2)==L_1**2 and round(((xtrade_uni[i][1]+L_2/math.sqrt(P_HIGH_2))*(ytrade_uni[i][1]+L_2*math.sqrt(P_LOW_2))),2)==L_2**2:  
        sectionx = np.linspace(xtrade_uni[i][0], 0, 500)
        section_dydx = -L_1**2/(sectionx+L_1/math.sqrt(P_HIGH_1))**2
        axes[13].fill_between(sectionx, section_dydx, alpha=0.5, color ='#afaffd')
        
        sectionx = np.linspace(Xmax2_uni, xtrade_uni[i][1], 500)
        section_dydx =-L_2**2/(sectionx+L_2/math.sqrt(P_HIGH_2))**2
        axes[13].fill_between(sectionx, section_dydx, alpha=0.5, color ='#fdbcbc') 
       
    
    axes[14].plot(xplot5_uni, dydxplot1_uni, color='blue')
    axes[14].plot(xplot2_uni, dydxplot2_uni, color='red')
    axes[14].scatter(
        [0, xplot5_uni[-1], xplot5_uni[0], xjointrade_uni[i][0], xjointrade_uni[i][1]],
        [dydxplot2_uni[0], dydxplot1_uni[-1], dydxplot1_uni[0], Yprice_curve_uni[i][0], Yprice_curve_uni[i][1]], 
        color=['orange', 'orange', 'green', 'red', 'blue'], 
        edgecolors=['red', 'blue', 'black', 'black', 'black'],
        s=80
    ) 
    ax14_elements = [
        mlines.Line2D([], [], color='red', markersize=10, label='dY/dX Curve 1'),
        mlines.Line2D([], [], color='blue', markersize=10, label='dY/dX Curve 2'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='red', markeredgecolor='black', markersize=10, label=f'Before Trade ({round(xjointrade_uni[i][0],2)} , {round(Yprice_curve_uni[i][0],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='blue', markeredgecolor='black',markersize=10, label=f'After Trade ({round(xjointrade_uni[i][1],2)} , {round(Yprice_curve_uni[i][1],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='orange',markeredgecolor='red', markersize=10, label=f'Price Bound 1 (0 , {round(dydxplot2_uni[0],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='orange',markeredgecolor='blue', markersize=10, label=f'Price Bound 2 ({round(xplot5_uni[-1],2)} , {round(dydxplot1_uni[-1],2)})'),
        mlines.Line2D([], [], color='none', marker ='o', markerfacecolor='green',markeredgecolor='black', markersize=10, label=f'P_Low1 = P_High2 ({round(xplot4_uni[-1],2)} , {round(dydxplot2_uni[-1],2)})'),
    ]
    axes[14].legend(handles=ax14_elements, loc='best', shadow=True, fancybox=True, title='Legend', title_fontsize='7', fontsize='7', framealpha=1)
    
    axes[14].set_xlabel("Token X Balance")
    axes[14].set_ylabel("dY / dX")
    axes[14].grid(True) 
    
    if xtrade_uni[i][1]-xtrade_uni[i][0]<0:  #round(Xmax1_bancor+Xmax2_bancor,2)
        arrow22 = FancyArrowPatch((xjointrade_uni[i][0], Yprice_curve_uni[i][0]), (xjointrade_uni[i][0], Yprice_curve_uni[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow23 = FancyArrowPatch((xjointrade_uni[i][0], Yprice_curve_uni[i][1]), (xjointrade_uni[i][1], Yprice_curve_uni[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow24 = FancyArrowPatch((xjointrade_uni[i][0], Yprice_curve_uni[i][0]), (xjointrade_uni[i][1], Yprice_curve_uni[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    elif xtrade_uni[i][1]-xtrade_uni[i][0]>0:
        arrow22 = FancyArrowPatch((xjointrade_uni[i][0], Yprice_curve_uni[i][0]), (xjointrade_uni[i][1], Yprice_curve_uni[i][0]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow23 = FancyArrowPatch((xjointrade_uni[i][1], Yprice_curve_uni[i][0]), (xjointrade_uni[i][1], Yprice_curve_uni[i][1]), arrowstyle='-|>,head_width=2,head_length=2', color='gold', linewidth=2)
        arrow24 = FancyArrowPatch((xjointrade_uni[i][0], Yprice_curve_uni[i][0]), (xjointrade_uni[i][1], Yprice_curve_uni[i][1]), arrowstyle='-|>,head_width=3,head_length=3', color='black', linewidth=3)
    
    axes[14].add_patch(arrow22)
    axes[14].add_patch(arrow23)
    axes[14].add_patch(arrow24)
    
    # Integral Area
    # sectionx = np.linspace(xjointrade[i][0], xjointrade[i][1], 500)
    if Xmax2_uni<xjointrade_uni[i][0] and Xmax2_uni<xjointrade_uni[i][1]:
        sectionx = np.linspace(xjointrade_uni[i][0], xjointrade_uni[i][1], 500)
        section_dydx = -L_1**2/((sectionx-Xmax2_uni)+L_1/math.sqrt(P_HIGH_1))**2
        axes[14].fill_between(sectionx, section_dydx, alpha=0.5, color ='#afaffd') 
    elif Xmax2_uni>xjointrade_uni[i][0] and Xmax2_uni>xjointrade_uni[i][1]:
        sectionx = np.linspace(xjointrade_uni[i][0], xjointrade_uni[i][1], 500)                    
        section_dydx =-L_2**2/(sectionx+L_2/math.sqrt(P_HIGH_2))**2
        axes[14].fill_between(sectionx, section_dydx, alpha=0.5, color ='#fdbcbc') 
    elif (Xmax2_uni>xjointrade_uni[i][0] and Xmax2_uni<xjointrade_uni[i][1]): ###
        sectionx = np.linspace(xjointrade_uni[i][0], Xmax2_uni, 500)
        section_dydx = -L_2**2/(sectionx+L_2/math.sqrt(P_HIGH_2))**2
        axes[14].fill_between(sectionx, section_dydx, alpha=0.5, color ='#fdbcbc')
        Xaux= xjointrade_uni[i][1]-Xmax2_uni
        sectionx = np.linspace(Xmax2_uni, xjointrade_uni[i][1], 500)
        section_dydx = -L_1**2/((sectionx-Xmax2_uni)+L_1/math.sqrt(P_HIGH_1))**2
        axes[14].fill_between(sectionx, section_dydx, alpha=0.5, color ='#afaffd') 
    elif (Xmax2_uni<xjointrade_uni[i][0] and Xmax2_uni>xjointrade_uni[i][1]):
        sectionx = np.linspace(Xmax2_uni,xjointrade_uni[i][0], 500)
        section_dydx = -L_1**2/((sectionx-Xmax2_uni)+L_1/math.sqrt(P_HIGH_1))**2
        axes[14].fill_between(sectionx, section_dydx, alpha=0.5, color ='#afaffd')
        sectionx = np.linspace(xjointrade_uni[i][1],Xmax2_uni, 500)
        section_dydx =-L_2**2/(sectionx+L_2/math.sqrt(P_HIGH_2))**2
        axes[14].fill_between(sectionx, section_dydx, alpha=0.5, color ='#fdbcbc') 
       
    # # First subplot with the actual trades
    plt.tight_layout()
    plt.show()