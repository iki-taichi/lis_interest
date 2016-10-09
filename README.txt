# lis_interest

unity側
stateはそのまま
actionにdegIntereset(degree of intereset; 興味度)を追加
FirstPersonControlerのFixedUpdateでの移動量の変更

python側
send_actionでdegInteresetも返すようにする
featureextractorの係数など微調整

q_net.pyに独自LSTMのクラスを追加
LSTMのクラスには別にoptimaizerを付けて最適化
    * update_prediction: self.Pを更新
    * calc_deg_intereset: self.Eを更新, 自身は0-1の興味度を返す
e_gredyでcalc_deg_intereset, update_prediction, LSTMエラーの積算
LSTMクラスはupdate_model_target内でbackprop
deg_interesetも返すようにする

問題点, feature_vecに対してそのままフルコネクトのLSTMをつけるとメモリーがかつかつ。
1 stepの計算時間がかなり長い...
convLSTMに変えるなど工夫が必要かも

