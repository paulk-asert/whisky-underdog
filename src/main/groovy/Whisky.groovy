/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import underdog.Underdog
import underdog.plots.charts.Chart
import underdog.plots.dsl.series.RadarSeries

import static underdog.plots.Options.create

def file = getClass().getResource('whisky.csv').file
def df = Underdog.df().read_csv(file).drop('RowID')

println df.shape()
println df.schema()

def features = df.columns - 'Distillery'
def plot = Underdog.plots()
plot.correlationMatrix(df[features]).show()

def selected = df[df['Fruity'] > 2 & df['Sweetness'] > 2]
println selected.shape()

plot.radar(
    features,
    [4] * features.size(),
    selected[features].toList()[0],
    selected['Distillery'][0]
).show()

def multiRadar = Chart.createGridOptions('Whisky flavor profiles', 'Somewhat sweet, somewhat fruity') +
create {
    radar {
        radius('50%')
        indicator(features.zip([4] * features.size())
            .collect { n, mx -> [name: n, max: mx] })
    }
    selected.toList().each { row ->
        series(RadarSeries) {
            data([[name: row[0], value: row[1..-1]]])
        }
    }
}.customize {
    legend {
        show(true)
    }
}
plot.show(multiRadar)

def ml = Underdog.ml()
def d = df[features] as double[][]
def clusters = ml.clustering.kMeans(d, nClusters: 3)
df['Cluster'] = clusters.toList()

println df.agg([Distillery:'count'])
    .by('Cluster')
    .sort_values(false, 'Cluster')
    .rename('Whisky Cluster Sizes')

println 'Clusters'
for (int i in clusters.toSet()) {
    println "$i:${df[df['Cluster'] == i]['Distillery'].join(', ')}"
}

def summary = df
    .agg(features.collectEntries{ f -> [f, 'mean']})
    .by('Cluster')
    .sort_values(false, 'Cluster')
    .rename('Flavour Centroids')

(summary.columns - 'Cluster').each { c ->
    summary[c] = summary[c](Double, Double) { it.round(3) }
}
println summary

def pca = ml.features.pca(d, 2)
def projected = pca.apply(d)
df['X'] = projected*.getAt(0)
df['Y'] = projected*.getAt(1)

plot.scatter(
    df['X'],
    df['Y'],
    df['Cluster'],
    'Whisky Clusters (kMeans)'
).show()

clusters = ml.clustering.agglomerative(d, nClusters: 3)
df['Cluster'] = clusters.toList()

println df.agg([Distillery:'count'])
    .by('Cluster')
    .rename('Whisky Cluster Sizes')

println 'Clusters'
for (int i in clusters.toSet()) {
    println "$i:${df[df['Cluster'] == i]['Distillery'].join(', ')}"
}

pca = ml.features.pca(d, 2)
projected = pca.apply(d)
df['X'] = projected*.getAt(0)
df['Y'] = projected*.getAt(1)

plot.scatter(
    df['X'],
    df['Y'],
    df['Cluster'],
    'Whisky Clusters (agglomerative)'
).show()
