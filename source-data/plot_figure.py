#!/usr/bin/env python3
"""Reproduce manuscript panels from the Source Data workbook.

Put this script and `Source Data.xlsx` in the same directory, then run:
    python plot_figures.py

Required Python packages:
    pandas numpy matplotlib scipy openpyxl

This script uses the final deposited source-data sheets as numerical input and
reuses the styling/layout choices from the latest revision-era plotting scripts.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

ROOT = Path(__file__).resolve().parent
FILE = ROOT / 'Source Data.xlsx'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
if not FILE.exists():
    raise FileNotFoundError(f'Could not find {FILE.name} next to this script.')

def rd(sheet):
    return pd.read_excel(FILE, sheet_name=sheet, engine='openpyxl')

def save(fig, name, dpi=300):
    for ext in ('svg','pdf','png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

C = {1:'#440154', 2:'#22A884', 3:'#D55E00'}


# -----------------------------------------------------------------------------
# Helper for panel labels
# -----------------------------------------------------------------------------
def panel_label(ax, label):
    ax.text(
        -0.12, 1.08, label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight='bold',
        ha='left',
        va='top'
    )

# -----------------------------------------------------------------------------
# Fig. 3 — PET violin and early/later associations
# Publication-sized typography
# -----------------------------------------------------------------------------

d3 = rd('Fig. 3')
d3['male'] = d3.sex.astype(str).str.lower().eq('male').astype(int)


def adjusted_means(frame, column):
    q = frame.dropna(subset=[column]).copy()

    y = q[column].to_numpy(float)
    age = q.age.to_numpy(float) - q.age.mean()

    X = np.column_stack([
        np.ones(len(q)),
        q.cluster.eq(2).astype(int),
        q.cluster.eq(3).astype(int),
        age,
        q.male
    ])

    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    resid = y - X @ beta
    df = len(q) - X.shape[1]

    cov = (
        (resid @ resid / df) *
        np.linalg.inv(X.T @ X)
    )

    crit = stats.t.ppf(.975, df)

    out = {}

    for g in (1, 2, 3):
        row = np.array([
            1,
            int(g == 2),
            int(g == 3),
            0,
            .5
        ])

        out[g] = (
            row @ beta,
            crit * np.sqrt(row @ cov @ row)
        )

    return out


def draw_violin_clusters(
    ax,
    frame,
    column,
    ylabel,
    brackets,
    bracket_fontsize=11
):
    ax.set_facecolor('white')
    ax.grid(False)

    rng = np.random.default_rng(42)
    adj = adjusted_means(frame, column)

    for g in (1, 2, 3):

        y = frame.loc[
            frame.cluster == g,
            column
        ].dropna().to_numpy()

        vp = ax.violinplot(
            [y],
            positions=[g],
            widths=.65,
            showextrema=False
        )

        for body in vp['bodies']:
            body.set_facecolor(C[g])
            body.set_edgecolor('none')
            body.set_alpha(.22)

        q1, med, q3 = np.quantile(
            y,
            [.25, .5, .75]
        )

        ax.plot(
            [g, g],
            [q1, q3],
            color=C[g],
            lw=5,
            solid_capstyle='butt',
            zorder=3
        )

        ax.scatter(
            g,
            med,
            s=18,
            c='white',
            edgecolor=C[g],
            lw=.9,
            zorder=4
        )

        ax.scatter(
            g + rng.uniform(-.09, .09, len(y)),
            y,
            c=C[g],
            s=19,
            alpha=.72,
            edgecolor='white',
            lw=.45,
            zorder=3
        )

        mean, ci = adj[g]

        ax.errorbar(
            g + .22,
            mean,
            yerr=ci,
            fmt='D',
            ms=4.1,
            mfc='black',
            mec='black',
            ecolor='black',
            elinewidth=.9,
            capsize=2.2,
            zorder=5
        )

    ax.set_xticks(
        [1, 2, 3],
        ['Cluster 1', 'Cluster 2', 'Cluster 3']
    )

    ax.set_xlim(.55, 3.55)
    ax.set_xlabel('Cluster')
    ax.set_ylabel(ylabel)

    lo, hi = ax.get_ylim()
    step = (hi - lo) * .12

    for j, (x1, x2, label) in enumerate(brackets):

        yy = hi + j * step

        ax.plot(
            [x1, x1, x2, x2],
            [
                yy - step * .15,
                yy,
                yy,
                yy - step * .15
            ],
            c='#333333',
            lw=.8,
            clip_on=False
        )

        ax.text(
            (x1 + x2) / 2,
            yy + step * .06,
            label,
            ha='center',
            va='bottom',
            fontsize=bracket_fontsize
        )

    ax.set_ylim(
        lo,
        hi + len(brackets) * step + step * .3
    )


# -----------------------------------------------------------------------------
# COMBINED FIGURE 3 — a/b/c
# -----------------------------------------------------------------------------

fig, axs = plt.subplots(
    1,
    3,
    figsize=(13.5, 4.8),
    sharey=False
)

# Increase typography specifically for this figure.
for ax in axs:
    ax.tick_params(
        axis='both',
        labelsize=11
    )
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)


# Panel a
draw_violin_clusters(
    axs[0],
    d3,
    'PET_gamma_over_k2a',
    'PET γ/k₂a in NAcc',
    [
        (1, 2, '**'),
        (1, 3, '*'),
        (2, 3, 'ns')
    ],
    bracket_fontsize=11
)

panel_label(
    axs[0],
    'a'
)


# Panels b and c
for ax, col, label, letter in zip(
    axs[1:],
    [
        'early_B2_minus_B1',
        'late_B4_minus_B3'
    ],
    [
        'Early change (Block 2 − Block 1)',
        'Later change (Block 4 − Block 3)'
    ],
    [
        'b',
        'c'
    ]
):

    ax.set_facecolor('white')
    ax.grid(False)

    for g in (1, 2, 3):

        sub = d3[
            (d3.cluster == g) &
            d3.PET_gamma_over_k2a.notna()
        ]

        ax.scatter(
            sub[col],
            sub.PET_gamma_over_k2a,
            c=C[g],
            s=27,
            alpha=.8,
            lw=0,
            label=f'Cluster {g}'
        )

    q = d3[
        [col, 'PET_gamma_over_k2a']
    ].dropna()

    slope, intercept, *_ = stats.linregress(
        q[col],
        q.PET_gamma_over_k2a
    )

    xx = np.linspace(
        q[col].min(),
        q[col].max(),
        100
    )

    ax.plot(
        xx,
        intercept + slope * xx,
        color='#333333',
        lw=1.4
    )

    ax.set_ylim(
        -.63,
        .63
    )

    ax.set_xlabel(
        label +
        '\nReward-to-cue shift\n'
        'NAcc BOLD beta weights',
        fontsize=12
    )

    ax.set_ylabel(
        'PET γ/k₂a in NAcc',
        fontsize=12
    )

    ax.tick_params(
        axis='both',
        labelsize=11
    )

    panel_label(
        ax,
        letter
    )


# Legend
axs[2].legend(
    frameon=False,
    fontsize=11,
    handletextpad=.5,
    borderaxespad=.4
)


fig.subplots_adjust(
    left=.07,
    right=.99,
    bottom=.21,
    top=.91,
    wspace=.28
)

save(
    fig,
    'Fig3'
)


# -----------------------------------------------------------------------------
# Fig. 4a — SHAPS by cluster
# -----------------------------------------------------------------------------
d4 = rd('Fig. 4a')
d4['male'] = d4.sex.astype(str).str.lower().eq('male').astype(int)

# -----------------------------------------------------------------------------
# Fig. 4b — valence ratings
# -----------------------------------------------------------------------------
d4b = rd('Fig. 4b')

def draw_valence(ax):
    ax.set_facecolor('white')
    ax.grid(False)

    for g in (1, 2, 3):
        sub = d4b[d4b.cluster == g]
        cols = [
            'large_minus_neutral_1',
            'large_minus_neutral_2',
            'large_minus_neutral_3'
        ]
        ys = sub[cols].mean().to_numpy()
        es = sub[cols].sem().to_numpy()

        ax.errorbar(
            [1, 2, 3],
            ys,
            yerr=es,
            color=C[g],
            marker='o',
            lw=2,
            capsize=3,
            label=f'Cluster {g} (n={len(sub)})'
        )

    ax.set_xticks([1, 2, 3])
    ax.set_xlabel('Assessment')
    ax.set_ylabel(
        'Valence rating: large reward minus neutral cue'
    )
    ax.legend(frameon=False, fontsize=8)

fig, ax = plt.subplots(figsize=(6, 4.5))
draw_valence(ax)
#save(fig, 'Fig4b')

# -----------------------------------------------------------------------------
# COMBINED FIGURE 4 — a/b in one row
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(
    1, 2,
    figsize=(11.5, 4.8)
)

draw_violin_clusters(
    axs[0],
    d4,
    'SHAPS',
    'SHAPS total score',
    [(1, 2, 'ns'), (2, 3, '*'), (1, 3, '*')]
)
panel_label(axs[0], 'a')

draw_valence(axs[1])
panel_label(axs[1], 'b')

fig.subplots_adjust(
    left=.08,
    right=.99,
    bottom=.14,
    top=.91,
    wspace=.32
)
save(fig, 'Fig4')

# -----------------------------------------------------------------------------
# FIGURE 5
# a = validation violin
# b = projected reward-/cue-dominant profiles
# c = odds ratios
# Publication-sized typography
# -----------------------------------------------------------------------------

d5 = rd('Fig. 5a-b').copy()

d5['shift'] = d5['raw_shift']

d5['sex_num'] = pd.to_numeric(
    d5.sex,
    errors='coerce'
).fillna(0)


labels5 = [
    ('p', 'HC'),
    ('p', 'MDD'),
    ('a', 'HC'),
    ('a', 'MDD')
]


X = np.column_stack([
    np.ones(len(d5)),
    d5.diagnosis.eq('MDD').astype(int),
    d5.treatment.eq('a').astype(int),
    (
        d5.diagnosis.eq('MDD') &
        d5.treatment.eq('a')
    ).astype(int),
    d5.age.to_numpy() - d5.age.mean(),
    d5.sex_num.to_numpy()
])


y = d5['shift'].to_numpy()

beta = np.linalg.lstsq(
    X,
    y,
    rcond=None
)[0]

resid = y - X @ beta

df = len(d5) - X.shape[1]

cov = (
    (resid @ resid / df) *
    np.linalg.inv(X.T @ X)
)

crit = stats.t.ppf(
    .975,
    df
)


colors = {
    'HC': '#2980B9',
    'MDD': '#C0392B'
}


# -----------------------------------------------------------------------------
# Significance bracket
# -----------------------------------------------------------------------------

def bracket(
    ax,
    x1,
    x2,
    y,
    label='*'
):

    h = .10

    ax.plot(
        [x1, x1, x2, x2],
        [y-h, y, y, y-h],
        color='#333333',
        lw=.9,
        clip_on=False
    )

    ax.text(
        (x1+x2)/2,
        y+.035,
        label,
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='normal'
    )


# -----------------------------------------------------------------------------
# PANEL a
# -----------------------------------------------------------------------------

def draw_fig5a(ax):

    ax.set_facecolor('white')

    ax.axhline(
        0,
        color='#555555',
        lw=.8
    )

    ax.grid(False)

    rng = np.random.default_rng(19)

    for i, (t, g) in enumerate(labels5):

        x = i + (i >= 2) * .4

        vals = d5.loc[
            (d5.treatment == t) &
            (d5.diagnosis == g),
            'shift'
        ].dropna().to_numpy()

        v = ax.violinplot(
            [vals],
            positions=[x],
            widths=.68,
            showextrema=False
        )

        for body in v['bodies']:

            body.set_facecolor(
                colors[g]
            )

            body.set_edgecolor(
                'none'
            )

            body.set_alpha(
                .22
            )

        q1, med, q3 = np.quantile(
            vals,
            [.25, .5, .75]
        )

        ax.plot(
            [x, x],
            [q1, q3],
            c=colors[g],
            lw=5,
            solid_capstyle='butt',
            zorder=3
        )

        ax.scatter(
            x,
            med,
            s=18,
            c='white',
            edgecolor=colors[g],
            lw=.9,
            zorder=4
        )

        ax.scatter(
            x + rng.uniform(
                -.09,
                .09,
                len(vals)
            ),
            vals,
            s=19,
            alpha=.72,
            c=colors[g],
            edgecolors='white',
            linewidths=.45,
            zorder=3
        )

        row = np.array([
            1,
            int(g == 'MDD'),
            int(t == 'a'),
            int(
                g == 'MDD' and
                t == 'a'
            ),
            0,
            .5
        ])

        mean = row @ beta

        ci = crit * np.sqrt(
            row @ cov @ row
        )

        ax.errorbar(
            x+.22,
            mean,
            yerr=ci,
            fmt='D',
            ms=4.1,
            mfc='black',
            mec='black',
            ecolor='black',
            elinewidth=.9,
            capsize=2.2,
            zorder=5
        )

    ax.set_xticks(
        [0, 1, 2.4, 3.4],
        ['HC', 'MDD', 'HC', 'MDD']
    )

    ax.tick_params(
        axis='both',
        labelsize=11
    )

    ax.text(
        .5,
        -.115,
        'Placebo',
        transform=ax.get_xaxis_transform(),
        ha='center',
        va='top',
        fontsize=11
    )

    ax.text(
        2.9,
        -.115,
        'Amisulpride',
        transform=ax.get_xaxis_transform(),
        ha='center',
        va='top',
        fontsize=11
    )

    ax.set_xlim(
        -.5,
        3.9
    )

    ax.set_ylim(
        -3.6,
        5.15
    )

    ax.set_ylabel(
        'Reward-to-cue shift\n'
        'NAcc BOLD beta weights',
        fontsize=12
    )

    # Significance brackets
    bracket(
        ax,
        0,
        1,
        3.68,
        '*'
    )

    bracket(
        ax,
        2.4,
        3.4,
        3.68,
        'ns'
    )

    bracket(
        ax,
        0,
        2.4,
        4.15,
        'ns'
    )

    bracket(
        ax,
        1,
        3.4,
        4.62,
        '*'
    )


# -----------------------------------------------------------------------------
# PANEL b
# -----------------------------------------------------------------------------

def draw_fig5b(axs):

    for ax, (arm, t) in zip(
        axs,
        [
            ('Amisulpride', 'a'),
            ('Placebo', 'p')
        ]
    ):

        s = d5[
            d5.treatment == t
        ]

        pairs = []

        for cl in [
            'C1',
            'C2'
        ]:

            z = s[
                s.cluster_pred == cl
            ].diagnosis.value_counts()

            pairs.append(
                (
                    int(z.get('HC', 0)),
                    int(z.get('MDD', 0))
                )
            )

        for x, (hc, mdd) in enumerate(
            pairs
        ):

            n = hc + mdd

            ax.bar(
                x,
                mdd/n,
                color='#377EB8',
                width=.74
            )

            ax.bar(
                x,
                hc/n,
                bottom=mdd/n,
                color='#E41A1C',
                width=.74
            )

        ax.set_xlim(
            -.5,
            1.5
        )

        ax.set_ylim(
            0,
            1.025
        )

        ax.set_xticks(
            [0, 1],
            [
                'Reward-dominant\nprofile',
                'Cue-dominant\nprofile'
            ]
        )

        # Increased from 8.5
        ax.tick_params(
            axis='x',
            labelsize=10
        )

        ax.tick_params(
            axis='y',
            labelsize=10
        )

        ax.set_title(
            arm,
            pad=10,
            fontsize=12
        )

        ax.grid(False)

    axs[0].set_ylabel(
        'Proportion',
        fontsize=12
    )

    # Remove duplicate y labels/ticks
    # from right subpanel
    axs[1].tick_params(
        axis='y',
        labelleft=False
    )


# -----------------------------------------------------------------------------
# PANEL c
# -----------------------------------------------------------------------------

or_df = rd(
    'Fig. 5c'
).set_index(
    'treatment'
)


def draw_fig5c(ax):

    ax.set_facecolor(
        'white'
    )

    ax.axvspan(
        .005,
        1,
        color='#E8DAEF',
        alpha=.25,
        zorder=0
    )

    ax.axvline(
        1,
        color='#555555',
        ls='--',
        lw=1
    )

    for y0, t, color, marker in [
        (
            1,
            'p',
            '#C0392B',
            'D'
        ),
        (
            0,
            'a',
            '#2980B9',
            'o'
        )
    ]:

        r = or_df.loc[t]

        or_ = float(
            r.conditional_OR
        )

        lo = float(
            r.CI95_lower
        )

        hi = float(
            r.CI95_upper
        )

        p = float(
            r.Fisher_p
        )

        ax.errorbar(
            or_,
            y0,
            xerr=[
                [or_-lo],
                [hi-or_]
            ],
            fmt=marker,
            c=color,
            ecolor=color,
            ms=7,
            capsize=4,
            lw=1.7,
            zorder=3
        )

        ax.text(
            9,
            y0,
            (
                f'OR = {or_:.2f} '
                f'[{lo:.2f}–{hi:.2f}]\n'
                f'p = {p:.3f}'
            ),
            color=color,
            va='center',
            fontsize=10
        )

    ax.set_xscale(
        'log'
    )

    ax.set_xlim(
        .005,
        100
    )

    ax.set_ylim(
        -.55,
        1.55
    )

    ax.set_xticks(
        [.01, .1, .5, 1, 2, 5],
        [
            '0.01',
            '0.1',
            '0.5',
            '1',
            '2',
            '5'
        ]
    )

    ax.set_yticks(
        [1, 0],
        [
            'Placebo',
            'Amisulpride'
        ]
    )

    ax.tick_params(
        axis='both',
        labelsize=10
    )

    ax.set_xlabel(
        'Odds ratio: cue-dominant vs reward-dominant profile '
        '(MDD relative to HC)',
        fontsize=11
    )

    ax.grid(False)


# =============================================================================
# COMBINED FIGURE 5
# =============================================================================

fig = plt.figure(
    figsize=(13.5, 7.4),
    facecolor='white'
)


# -----------------------------------------------------------------------------
# Manual positions
# [left, bottom, width, height]
# -----------------------------------------------------------------------------

top_bottom = .53
top_height = .39


# Panel a
ax_a = fig.add_axes([
    .08,
    top_bottom,
    .40,
    top_height
])


# Panel b1
ax_b1 = fig.add_axes([
    .57,
    top_bottom,
    .17,
    top_height
])


# Panel b2
ax_b2 = fig.add_axes([
    .77,
    top_bottom,
    .17,
    top_height
], sharey=ax_b1)


# Panel c
ax_c = fig.add_axes([
    .17,
    .11,
    .68,
    .29
])


# -----------------------------------------------------------------------------
# Draw panels
# -----------------------------------------------------------------------------

draw_fig5a(
    ax_a
)

draw_fig5b([
    ax_b1,
    ax_b2
])

draw_fig5c(
    ax_c
)


# -----------------------------------------------------------------------------
# Panel b legend
# -----------------------------------------------------------------------------

legend_handles = [
    Patch(
        facecolor='#E41A1C',
        label='HC'
    ),
    Patch(
        facecolor='#377EB8',
        label='MDD'
    )
]


fig.legend(
    handles=legend_handles,
    loc='center',
    bbox_to_anchor=(
        .755,
        .465
    ),
    ncol=2,
    frameon=False,
    title='Diagnosis',
    fontsize=10,
    title_fontsize=10,
    handlelength=1.8,
    columnspacing=1.5
)


# -----------------------------------------------------------------------------
# Panel labels
# -----------------------------------------------------------------------------

label_fontsize = 16


# a
fig.text(
    .045,
    .94,
    'a',
    fontsize=label_fontsize,
    fontweight='bold',
    ha='left',
    va='bottom'
)


# b
fig.text(
    .535,
    .94,
    'b',
    fontsize=label_fontsize,
    fontweight='bold',
    ha='left',
    va='bottom'
)


# c
fig.text(
    .135,
    .415,
    'c',
    fontsize=label_fontsize,
    fontweight='bold',
    ha='left',
    va='bottom'
)


# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------

save(
    fig,
    'Fig5'
)

# -----------------------------------------------------------------------------
# Supplementary Fig. 2 — MID cue/outcome collinearity
# -----------------------------------------------------------------------------
a=rd('Supplementary Fig. 2a').rename(columns={'ID':'Subject'});b=rd('Supplementary Fig. 2b').rename(columns={'ID':'Subject'})
rng=np.random.default_rng(20260924);fig,axs=plt.subplots(1,2,figsize=(11.5,5.5),gridspec_kw={'width_ratios':[1.6,1]});colors2={'Large':'#7B3294','Reward':'#7B3294','Neutral':'#008837'}
def distribution(ax,x,y,color,width=.55):
    y=np.asarray(y);v=ax.violinplot(y,positions=[x],widths=width,showextrema=False,bw_method=.35)
    for body in v['bodies']:body.set_facecolor(color);body.set_edgecolor('none');body.set_alpha(.23)
    ax.scatter(x+rng.uniform(-width*.16,width*.16,len(y)),y,s=19,color=color,alpha=.65,edgecolors='white',linewidths=.35,zorder=3);q1,med,q3=np.quantile(y,[.25,.5,.75]);ax.plot([x+width*.30]*2,[q1,q3],color='black',lw=2.1,zorder=5);ax.scatter([x+width*.30],[med],marker='D',s=24,color='black',zorder=6)
for ax in axs:ax.spines[['top','right']].set_visible(False);ax.tick_params(direction='out',length=4,width=1);ax.axhline(0,color='#999999',ls='--',lw=.8,zorder=0);ax.set_ylim(-.05,.405);ax.set_yticks(np.arange(0,.41,.1))
for block in range(1,5):
    for cond,offset in [('Large',-.19),('Neutral',.19)]:distribution(axs[0],block+offset,a.loc[(a.Block==block)&(a.Condition==cond),'Correlation'],colors2[cond],.34)
axs[0].set(xlim=(.45,4.55),xticks=[1,2,3,4],xlabel='Learning block',ylabel="Pearson’s r");axs[0].set_title('Modified MID: cue–outcome',loc='left',pad=30);axs[0].text(0,1.025,'Primary MDD sample · n = 57',transform=axs[0].transAxes,color='#555555',fontsize=10.5)
for x,cond in [(1,'Reward'),(2,'Neutral')]:distribution(axs[1],x,b.loc[b.Condition==cond,'Correlation'],colors2[cond],.65)
axs[1].set(xlim=(.45,2.55),xticks=[1,2],xticklabels=['Reward','Neutral'],xlabel='Condition',ylabel="Pearson’s r");axs[1].set_title('Classic MID: cue–outcome',loc='left',pad=30);axs[1].text(0,1.025,'Independent MDD–HC sample · n = 89',transform=axs[1].transAxes,color='#555555',fontsize=10)
for letter,ax in zip('AB',axs):ax.text(-.15,1.15,letter,transform=ax.transAxes,fontweight='bold',fontsize=17)
handles=[Line2D([],[],marker='o',ls='',color=colors2['Reward'],label='Reward (large reward in A)',markersize=6),Line2D([],[],marker='o',ls='',color=colors2['Neutral'],label='Neutral',markersize=6),Line2D([],[],marker='D',ls='-',color='black',label='Median and interquartile range',markersize=5)]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.025),ncol=3,frameon=False,fontsize=11);fig.subplots_adjust(left=.09,right=.98,top=.79,bottom=.23,wspace=.32);save(fig,'Supplementary_Fig2')

# -----------------------------------------------------------------------------
# Supplementary Fig. 15 — reward vs penalty validation (latest loss style)
# -----------------------------------------------------------------------------
dl=rd('Supplementary Fig. 15').copy();dl['Reward']=dl.reward_shift_winsor;dl['Loss']=dl.penalty_shift_winsor;dl['age_c']=dl.age-dl.age.mean();dl['sex_num']=pd.to_numeric(dl.sex,errors='coerce').fillna(0)
X=np.column_stack([np.ones(len(dl)),(dl.diagnosis=='HC').astype(float),(dl.treatment=='a').astype(float),((dl.diagnosis=='HC')&(dl.treatment=='a')).astype(float),dl.age_c,dl.sex_num])
class Model:
    def __init__(self,y):self.beta=np.linalg.lstsq(X,y,rcond=None)[0];self.df=len(y)-X.shape[1];self.cov=np.linalg.inv(X.T@X)*np.sum((y-X@self.beta)**2)/self.df
    def t(self,r):
        e=r@self.beta;se=np.sqrt(r@self.cov@r);return e,se,2*stats.t.sf(abs(e/se),self.df)
def row(g):
    hc=float(g[0]=='HC');a=float(g[1]=='a');return np.array([1,hc,a,hc*a,0,dl.sex_num.mean()])
models={v:Model(dl[v].to_numpy()) for v in ['Reward','Loss']};groups=[('HC','p'),('MDD','p'),('HC','a'),('MDD','a')];positions=[0,1,2.5,3.5];colors3={'HC':'#2980B9','MDD':'#C0392B'};rng=np.random.default_rng(20260922)
lo=np.floor(dl[['Reward','Loss']].min().min())-.7;hi=np.ceil(dl[['Reward','Loss']].max().max())+.7;fig,axes=plt.subplots(1,2,figsize=(13.5,8.2),sharey=True)
for ax,v,letter,title in zip(axes,['Reward','Loss'],['a','b'],['Reward trials','Penalty trials']):
    m=models[v];ax.grid(False)
    for x,g in zip(positions,groups):
        yy=dl.loc[(dl.diagnosis==g[0])&(dl.treatment==g[1]),v].to_numpy();c=colors3[g[0]];vio=ax.violinplot(yy,positions=[x],widths=.82,showextrema=False)
        for body in vio['bodies']:body.set_facecolor(c);body.set_edgecolor('none');body.set_alpha(.22)
        q1,med,q3=np.percentile(yy,[25,50,75]);ax.plot([x,x],[q1,q3],color=c,lw=8,solid_capstyle='butt',zorder=3);jit=rng.uniform(-.12,.12,len(yy));ax.scatter(x+jit-.065,yy,s=35,facecolor=c,edgecolor='white',alpha=.7,linewidth=.8,zorder=4);ax.scatter([x],[med],s=24,facecolor='white',edgecolor='none',zorder=5)
        mean,se,_=m.t(row(g));crit=stats.t.ppf(.975,m.df);ax.errorbar(x+.28,mean,yerr=crit*se,fmt='D',color='black',ms=6,capsize=4,lw=1.4,zorder=6)
    for i,j,level in [(0,1,0),(2,3,0),(0,2,1),(1,3,2)]:
        _,_,pv=m.t(row(groups[i])-row(groups[j]));yb=hi-.1+level*.65;ax.plot([positions[i],positions[i],positions[j],positions[j]],[yb,yb+.12,yb+.12,yb],color='#333333',lw=1);ax.text((positions[i]+positions[j])/2,yb+.17,('***' if pv<.001 else '**' if pv<.01 else '*' if pv<.05 else 'ns'),ha='center',fontsize=10)
    ax.axhline(0,color='#333333',ls=(0,(5,4)),lw=1);ax.set_ylim(lo,hi+2);ax.set_xlim(-.5,4.05);ax.set_xticks(positions,['HC','MDD','HC','MDD'],fontsize=10);ax.text(.5,-.13,'Placebo',ha='center',transform=ax.get_xaxis_transform(),weight='bold');ax.text(3,-.13,'Amisulpride',ha='center',transform=ax.get_xaxis_transform(),weight='bold');ax.set_title(title,pad=37,weight='bold');ax.text(-.075,1.15,letter,transform=ax.transAxes,fontsize=20,weight='bold');ax.tick_params(axis='both',length=0,labelcolor='#4D4D4D')
axes[0].set_ylabel('Reward-to-cue shift\nNAcc BOLD beta weights');fig.subplots_adjust(left=.09,right=.985,bottom=.23,top=.80,wspace=.12);save(fig,'Supplementary_Fig15')




# -----------------------------------------------------------------------------
# Supplementary Fig. 11 — PET cluster differences:
# original versus QC-adjusted clustering
# -----------------------------------------------------------------------------
s11 = rd('Supplementary Fig. 11').copy()

contrast_order_11 = [
    'Cluster 2 − Cluster 1',
    'Cluster 3 − Cluster 1',
    'Cluster 3 − Cluster 2'
]
analysis_order_11 = ['Original clustering', 'QC-adjusted clustering']
style11 = {
    'Original clustering': dict(color='#333333', marker='o'),
    'QC-adjusted clustering': dict(color='#1F77B4', marker='s')
}

fig, ax = plt.subplots(figsize=(9.0, 4.2))
ax.set_facecolor('white')
ax.grid(False)
ax.axvline(0, color='#999999', ls='--', lw=1)

ybase = np.array([2, 1, 0], dtype=float)
offset11 = {
    'Original clustering': 0.10,
    'QC-adjusted clustering': -0.10
}

for analysis in analysis_order_11:
    sub = s11[s11.analysis == analysis].set_index('contrast')
    st = style11[analysis]

    for i, contrast in enumerate(contrast_order_11):
        r = sub.loc[contrast]
        y = ybase[i] + offset11[analysis]
        est = float(r.estimate)
        lo = float(r.CI_low)
        hi = float(r.CI_high)
        q = float(r.p_FDR)

        ax.errorbar(
            est, y,
            xerr=[[est-lo], [hi-est]],
            fmt=st['marker'],
            color=st['color'],
            ecolor=st['color'],
            ms=6,
            capsize=3,
            lw=1.4,
            zorder=3
        )

        ax.text(
            hi + .012, y,
            f'q={q:.3f}',
            va='center',
            ha='left',
            fontsize=9,
            color=st['color']
        )

ax.set_yticks(ybase, contrast_order_11)
ax.set_xlim(-.35, .51)
ax.set_xlabel(r'Adjusted difference in NAcc dopamine release ($\gamma/k_{2a}$)')
ax.set_title(
    'PET cluster differences are robust to QC-adjusted clustering',
    fontsize=13,
    fontweight='bold',
    pad=8
)

handles11 = [
    Line2D([0], [0], marker='o', color='#333333',
           lw=1.4, label='Original clustering', markersize=6),
    Line2D([0], [0], marker='s', color='#1F77B4',
           lw=1.4, label='QC-adjusted clustering', markersize=6)
]
ax.legend(
    handles=handles11,
    loc='lower center',
    bbox_to_anchor=(.5, -.34),
    ncol=2,
    frameon=False
)
fig.subplots_adjust(left=.23, right=.96, bottom=.27, top=.87)
save(fig, 'Supplementary_Fig11')


# -----------------------------------------------------------------------------
# Supplementary Fig. 12 — winsorization sensitivity analyses
# -----------------------------------------------------------------------------
s12ab = rd('Supplementary Fig. 12a-b').copy()
s12c = rd('Supplementary Fig. 12c').copy()
s12d = rd('Supplementary Fig. 12d').copy()
s12e = rd('Supplementary Fig. 12e').copy()

reward_order = ['neutral', 'small', 'medium', 'large']
reward_labels = {
    'neutral': 'Neutral',
    'small': 'Small',
    'medium': 'Medium',
    'large': 'Large'
}
reward_cols = {
    'neutral': '#8E8E8E',
    'small': '#56B4E9',
    'medium': '#E69F00',
    'large': '#6A3D9A'
}

def draw_s12_replacement_bars(ax, phase, letter, title):
    q = s12ab[s12ab.phase == phase].copy()
    width = .18
    offsets = np.array([-1.5, -.5, .5, 1.5]) * width

    for j, reward in enumerate(reward_order):
        z = q[q.reward == reward].set_index('block').reindex([1, 2, 3, 4])
        xs = np.arange(1, 5) + offsets[j]
        heights = z['percent'].to_numpy(float)
        counts = z['n_replaced'].to_numpy(int)

        ax.bar(
            xs, heights,
            width=width,
            color=reward_cols[reward],
            edgecolor='white',
            linewidth=.5,
            label=reward_labels[reward]
        )

        for x, h, n in zip(xs, heights, counts):
            ax.text(
                x, h + .18, str(n),
                ha='center',
                va='bottom',
                fontsize=8
            )

    ax.set_xlim(.45, 4.55)
    ax.set_ylim(0, 12)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel('MID block')
    ax.set_ylabel('Values replaced (%)')
    ax.set_title(f'{letter} {title}', loc='left', fontweight='bold', fontsize=12)
    ax.grid(False)

# Assignment-agreement matrix and ARI computed from the deposited matrix
matrix12 = s12c.iloc[:, 1:4].to_numpy(dtype=int)
n12 = int(matrix12.sum())
unchanged12 = int(np.trace(matrix12))
row_sums12 = matrix12.sum(axis=1)
col_sums12 = matrix12.sum(axis=0)

def choose2(x):
    x = np.asarray(x, dtype=float)
    return x * (x - 1) / 2

sum_cells = choose2(matrix12).sum()
sum_rows = choose2(row_sums12).sum()
sum_cols = choose2(col_sums12).sum()
total_pairs = choose2(n12)
expected_index = (sum_rows * sum_cols) / total_pairs
max_index = .5 * (sum_rows + sum_cols)
ari12 = (sum_cells - expected_index) / (max_index - expected_index)

def draw_s12_matrix(ax):
    im = ax.imshow(matrix12, cmap='Blues', vmin=0, vmax=matrix12.max())

    for i in range(3):
        for j in range(3):
            val = matrix12[i, j]
            ax.text(
                j, i, str(val),
                ha='center', va='center',
                fontsize=12,
                color='white' if val > matrix12.max() * .5 else 'black'
            )

    ax.set_xticks([0, 1, 2], ['Cluster 1', 'Cluster 2', 'Cluster 3'])
    ax.set_yticks([0, 1, 2], ['Cluster 1', 'Cluster 2', 'Cluster 3'])
    ax.set_xlabel('Original cluster')
    ax.set_ylabel('Unwinsorized cluster')
    ax.set_title('C Assignment agreement', loc='left',
                 fontweight='bold', fontsize=12)
    ax.tick_params(length=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(
        .5, -.22,
        f'{unchanged12}/{n12} unchanged ({100*unchanged12/n12:.1f}%); '
        f'ARI = {ari12:.3f}',
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=8.5,
        fontweight='bold'
    )

def draw_s12_forest(ax, frame, contrasts, letter, title,
                    xlabel, xlim=None):
    ybase = np.arange(len(contrasts))[::-1].astype(float)
    offsets = {'Winsorized': .10, 'Unwinsorized': -.10}
    styles = {
        'Winsorized': dict(color='#333333', marker='o'),
        'Unwinsorized': dict(color='#0072B2', marker='s')
    }

    ax.axvline(0, color='#999999', ls='--', lw=.9)

    for analysis in ['Winsorized', 'Unwinsorized']:
        sub = frame[frame.analysis == analysis].set_index('contrast')
        st = styles[analysis]

        for i, contrast in enumerate(contrasts):
            r = sub.loc[contrast]
            est = float(r.estimate)
            lo = float(r.CI_low)
            hi = float(r.CI_high)
            y = ybase[i] + offsets[analysis]

            ax.errorbar(
                est, y,
                xerr=[[est-lo], [hi-est]],
                fmt=st['marker'],
                color=st['color'],
                ecolor=st['color'],
                ms=4.5,
                capsize=2.5,
                lw=1.2
            )

    ax.set_yticks(ybase, contrasts)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(f'{letter} {title}', loc='left',
                 fontweight='bold', fontsize=12)
    ax.grid(False)

    handles = [
        Line2D([0], [0], marker='o', color='#333333',
               lw=1.2, label='Winsorized', markersize=5),
        Line2D([0], [0], marker='s', color='#0072B2',
               lw=1.2, label='Unwinsorized', markersize=5)
    ]
    ax.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(.5, -.05),
        ncol=2,
        frameon=False,
        fontsize=7
    )

# Manual layout revised to prevent overlap/cropping:
# A/B top; C/D middle; E centered across the bottom.
# Extra vertical space is reserved below C/D and E for legends/captions.
fig = plt.figure(figsize=(11.2, 14.6), facecolor='white')

ax12a = fig.add_axes([.07, .755, .41, .195])
ax12b = fig.add_axes([.54, .755, .41, .195], sharey=ax12a)

ax12c = fig.add_axes([.10, .455, .35, .205])
ax12d = fig.add_axes([.61, .455, .34, .205])

ax12e = fig.add_axes([.20, .125, .72, .205])

draw_s12_replacement_bars(ax12a, 'cue', 'A', 'Cue')
draw_s12_replacement_bars(ax12b, 'outcome', 'B', 'Outcome')
ax12b.set_ylabel('')
ax12b.tick_params(axis='y', labelleft=False)

# Shared legend for A/B — placed in the open band between rows.
handles_ab = [
    Patch(facecolor=reward_cols[r], label=reward_labels[r])
    for r in reward_order
]
fig.legend(
    handles=handles_ab,
    loc='center',
    bbox_to_anchor=(.51, .705),
    ncol=4,
    frameon=False,
    fontsize=9,
    handlelength=1.8,
    columnspacing=2.0
)

draw_s12_matrix(ax12c)

draw_s12_forest(
    ax12d,
    s12d,
    ['C2 − C1', 'C3 − C1', 'C3 − C2'],
    'D',
    'PET cluster comparisons',
    r'Adjusted PET difference ($\gamma/k_{2a}$)' +
    '\nwith 95% confidence interval',
    xlim=(-.29, .48)
)
ax12d.set_yticklabels([
    'Cluster 2 − Cluster 1',
    'Cluster 3 − Cluster 1',
    'Cluster 3 − Cluster 2'
])

# Remove the helper legend from D and replace it below the x-axis,
# with enough clearance from the two-line x-axis title.
if ax12d.get_legend() is not None:
    ax12d.get_legend().remove()

forest_handles = [
    Line2D([0], [0], marker='o', color='#333333',
           lw=1.2, label='Winsorized', markersize=5),
    Line2D([0], [0], marker='s', color='#0072B2',
           lw=1.2, label='Unwinsorized', markersize=5)
]

ax12d.legend(
    handles=forest_handles,
    loc='upper center',
    bbox_to_anchor=(.5, -.29),
    ncol=2,
    frameon=False,
    fontsize=7,
    handlelength=1.6,
    columnspacing=1.5,
    borderaxespad=0
)
ax12d.xaxis.labelpad = 7

draw_s12_forest(
    ax12e,
    s12e,
    ['Early (B2 − B1)', 'Late (B4 − B3)', 'Overall (B4 − B1)'],
    'E',
    'Continuous fMRI–PET associations',
    'Age/sex-adjusted PET regression coefficient' +
    '\nwith 95% confidence interval',
    xlim=(-.017, .135)
)
ax12e.set_yticklabels([
    'Early: B2 − B1',
    'Late: B4 − B3',
    'Overall: B4 − B1'
])

# As in D, place the legend well below the two-line x-axis title.
if ax12e.get_legend() is not None:
    ax12e.get_legend().remove()

ax12e.legend(
    handles=forest_handles,
    loc='upper center',
    bbox_to_anchor=(.5, -.30),
    ncol=2,
    frameon=False,
    fontsize=8,
    handlelength=1.7,
    columnspacing=1.8,
    borderaxespad=0
)
ax12e.xaxis.labelpad = 7

save(fig, 'Supplementary_Fig12')


print(f'Finished. Figures saved to: {OUT}')
