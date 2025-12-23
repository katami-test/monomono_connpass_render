import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from amplify import VariableGenerator, Model, FixstarsClient, solve
from datetime import timedelta

#テキスト表示
st.title('都市を選択して巡回セールスマン問題を実施')
st.write('巡回する都市を選択してFixstars Amplifyで最短ルートを計算')

#サイドバー設定
with st.sidebar:
    st.write('計算のパラメーター設定')
    time_calc = st.number_input('タイムアウト時間',max_value = 10000, value = 1000, step = 1000)

#都市を設定
cities = {
    'A': [23, 56], 'B': [12, 24], 'C': [36, 72], 'D': [9, 53], 'E': [61, 55],
    'F': [47, 19], 'G': [33, 68], 'H': [28, 44], 'I': [71, 32], 'J': [15, 77],
    'K': [52, 26], 'L': [39, 64], 'M': [19, 81], 'N': [48, 23], 'O': [63, 57],
    'P': [26, 49], 'Q': [34, 66], 'R': [54, 21], 'S': [45, 78], 'T': [29, 31],
    'U': [67, 42], 'V': [37, 59], 'W': [41, 73], 'X': [12, 65], 'Y': [55, 36],
    'Z': [22, 48]}

cities_list = list(cities.keys())

col1, col2 = st.columns(2) #横並びレイアウトを作成

with col1:
    #都市を選択
    selected_cities = st.multiselect('都市を選択',cities_list,default = cities_list)
with col2:
    import numpy as np
    import matplotlib.pyplot as plt

    if selected_cities:  # 都市が選択されている場合
        #選択した都市の座標をndarrayにする
        selected_coords = np.array([cities[pref] for pref in selected_cities])
        N = selected_coords.shape[0]  # 都市数

        # 都市の座標をグラフで表示
        plt.figure(figsize=(7, 7))
        plt.plot(selected_coords[:, 0], selected_coords[:, 1], 'o')
        st.pyplot(plt)
    else:  # 都市が選択されていない場合
        selected_coords = np.array([])  # 空のndarrayをセット
        N = 0

        # 空のグラフを表示
        plt.figure(figsize=(7, 7))
        plt.title('No selected')
        st.pyplot(plt)

button1 = st.button('SQAを実行') #ボタンを作成
if button1:
    # 都市間の距離行列を作成
    x = selected_coords[:, 0]
    y = selected_coords[:, 1]
    d = np.sqrt((x[:, np.newaxis] - x[np.newaxis, :]) ** 2 +(y[:, np.newaxis] - y[np.newaxis, :]) ** 2)

    # 変数を生成
    gen = VariableGenerator()
    q = gen.array('Binary', shape=(N + 1, N))
    q[N, :] = q[0, :]

    # コスト関数

    cost = sum(sum(sum(d[i, j] * q[t, i] * q[t + 1, j] for j in range(N)) for i in range(N)) for t in range(N))

    # 罰金項

    lam = np.amax(d)
    penalty1 = lam * sum((sum(q[i, j] for j in range(N)) - 1) ** 2 for i in range(N)) # 行の合計が1
    penalty2 = lam * sum((sum(q[i, j] for i in range(N)) - 1) ** 2 for j in range(N)) # 列の合計が1

    model = Model(cost + penalty1 + penalty2)

    # ソルバーの設定
    client = FixstarsClient()
    client.token = '' # アクセストークンを入力
    client.parameters.outputs.duplicate = True
    client.parameters.outputs.num_outputs = 0
    client.parameters.timeout = timedelta(milliseconds=1000)

    # 結果の取得
    result = solve(model, client)
    if len(result) == 0:
        raise RuntimeError('At least one of the constraints is not satisfied.')
    num_runs = len(result)
    q_values = q.evaluate(result.best.values)
    route = np.where(np.array(q_values) == 1)[1]

    # 結果を表示
    num_runs = len(result)
    total_distance = result.best.objective
    st.write(f'計算回数:{num_runs}')  
    st.write(f'総距離:{total_distance}')

    # グラフにプロット
    route = np.where(np.array(q_values) == 1)[1]

    x = [i[0] for i in selected_coords]
    y = [i[1] for i in selected_coords]
    plt.figure(figsize=(7, 7))
    plt.xlabel('x')
    plt.ylabel('y')

    for i in range(N):
        r = route[i]
        n = route[i + 1]
        plt.plot([x[r], x[n]], [y[r], y[n]], 'b-')
    plt.plot(x, y, 'ro')
    st.pyplot(plt)