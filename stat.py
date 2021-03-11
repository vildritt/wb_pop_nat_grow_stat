#!/usr/bin/env python3

import os

import wbdata as wbd
import pandas as pd
import numpy as np

import bokeh.io as bio
import bokeh.models as bm
import bokeh.plotting as bp
import geopandas as gpd
import colorcet

year_from = 1990
year_to = 2021
value_to_draw = 'natural_grow'  # natural_grow net_migration total_grow

region_name = 'Europe'

countries = wbd.get_country()
country_to_id = {i["name"]: i["id"] for i in countries}

indicators = {
    "SP.POP.TOTL": "pop",
    "SP.DYN.CDRT.IN": "dr",
    "SP.DYN.CBRT.IN": "br",
}

CACHE_FILE = "data.gz"

if os.path.isfile(CACHE_FILE):
    print("Load from cache..")
    df = pd.read_pickle(CACHE_FILE)
else:
    print("Download..")
    df = wbd.get_dataframe(indicators, convert_date=True)
    df.to_pickle(CACHE_FILE)

df["births"] = df["pop"] * df["br"] / 1000.0
df["deaths"] = df["pop"] * df["dr"] / 1000.0

df.sort_index(0, level=[0, 1], inplace=True)
df.index = df.index.set_levels(df.index.levels[1].year, level=1)
df = df.rename_axis(index=['country', 'year'])

df["next_pop"] = df["pop"].shift(-1)
df["natural_grow"] = df["births"] - df["deaths"]
df["net_migration"] = df["next_pop"] - (df["pop"] + df["natural_grow"])

df = df.query('year >= {} and (year <= {})'.format(year_from, year_to))
df = df.dropna()

df_r = pd.DataFrame([], columns=[
    "year_from", "year_to", "pop", "natural_grow", "net_migration", "total_grow", "gap_count"])

n_r = df.shape[0]
c_prev = ""
y_prev = 0
y_init = 0
p_init = 0
t_ng = 0
t_nm = 0
t_cnt_pass = 0


def add_result():
    global df_r
    if p_init == 0:
        return
    ng = t_ng / p_init * 100.0
    nm = t_nm / p_init * 100.0
    tot = ng + nm
    row_data = {
        "year_from": y_init,
        "year_to": y_prev,
        "pop": p_init,
        "natural_grow": ng,
        "net_migration": nm,
        "total_grow": tot,
        "gap_count": t_cnt_pass
    }
    s = pd.Series(row_data, name=c_prev)
    df_r = df_r.append(s)


for i in range(n_r):
    r = df.iloc[i]
    c, y = r.name
    population, next_population, natural_grow, net_migration = \
        r["pop"], r["next_pop"], \
        r["natural_grow"], r["net_migration"]
    if c != c_prev:
        add_result()
        y_init = y
        p_init = population
        t_ng = 0
        t_nm = 0
        t_cnt_pass = 0
        c_prev = c
    else:
        if y > (y_prev+1):
            t_cnt_pass += 1
    t_ng += natural_grow
    t_nm += net_migration
    y_prev = y

add_result()

df_r.sort_values(value_to_draw, inplace=True)
print(df_r.to_string())

# plot

world_region = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world_region["iso_a3"].iloc[43] = "FRA"
world_region["iso_a3"].iloc[21] = "NOR"

c_id_to_abr = world_region["iso_a3"].to_dict()
c_abr_to_id = {v: int(k) for k, v in c_id_to_abr.items()}


df_r.index = df_r.index.map(country_to_id).map(c_abr_to_id)
df_r = df_r.loc[~np.isnan(df_r.index)]
df_r.index = df_r.index.map(lambda x: int(x))
df_r.sort_index(0, inplace=True)

world_region[value_to_draw] = df_r[value_to_draw]
world_region["text"] = df_r[value_to_draw].apply(lambda x: "{:.2f} %".format(x))
c = world_region["geometry"].centroid
world_region["x"] = c.x
world_region["y"] = c.y

coord_fixes = {
    "RUS": (35, 60),
    "FRA": (2, 46),
    "GBR": (-1, 52),
    "FIN": (26, 62),
    "NOR": (10, 60),
    "PRT": (-7, 38),
    "ITA": (10, 45),
}

for c_id, coord in coord_fixes.items():
    cid = c_abr_to_id[c_id]
    world_region["x"].loc[cid] = coord[0]
    world_region["y"].loc[cid] = coord[1]

if region_name:
    world_region = world_region.loc[world_region['continent'] == region_name]

plot_df = bm.GeoJSONDataSource(geojson=world_region.to_json())
p = bp.figure(
    title=value_to_draw, x_range=(-22, 40), y_range=(35, 72))
map_palette = colorcet.bkr

m = 20
color_mapper = bm.LinearColorMapper(palette=map_palette, low=-m, high=m)
p.patches(
    'xs', 'ys',
    fill_alpha=1.0,
    fill_color={
        'field': value_to_draw,
        'transform': color_mapper
    },
    line_color='black',
    line_width=1.0,
    source=plot_df)

color_bar = bm.ColorBar(
    color_mapper=color_mapper, ticker=bm.LogTicker(),
    label_standoff=12, border_line_color=None, location=(0, 0))
p.add_layout(color_bar, 'right')
p.add_tools(bm.WheelZoomTool())
labels = bm.LabelSet(
    x='x', y='y', text='text', source=plot_df,
    text_align='center',
    text_color='#FFFFFF',
    text_font_size={'value': '10px'},
    render_mode='canvas',
)
p.add_layout(labels)
bio.show(p)

try:
    bio.export_png(p, filename="{}.png".format(value_to_draw))
except:
    pass