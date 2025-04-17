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

def file = new File(getClass().classLoader.getResource('whiskey.csv').file)
def df = Underdog.df().read_csv(file.path).drop('RowID')

println df.shape()

println df.schema()

def features = df.columns - 'Distillery'

def plot = Underdog.plots().correlationMatrix(df[features])

plot.show()

plot = Underdog
    .plots()
    .radar(
        features,
        [4] * features.size(),
        (df[features] as double[][])[0].toList(),
        df['Distillery'][0]
    )
plot.show()

def ml = Underdog.ml()
def data = df[features] as double[][]

def clusters = ml.clustering.kMeans(data, nClusters: 3)
df['Cluster'] = clusters.toList()

println 'Clusters'
for (int i in clusters.toSet()) {
    println "$i:${df[df['Cluster'] == i]['Distillery'].join(', ')}"
}

def pca = ml.features.pca(data, 2)
def projected = pca.apply(data)

df['X'] = projected*.getAt(0)
df['Y'] = projected*.getAt(1)

plot = Underdog.plots()
    .scatter(
        df['X'],
        df['Y'],
        df['Cluster'],
        'Whiskey Clusters')

plot.show()
