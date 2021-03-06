# Deep Furiganaを機械学習で自動でふる

注：今回、JSAI2017において、立命館大学の学生が発表した論文が、一部の小説家の方々の批判を浴びたそうですが、この内容はgithubにて炎上前から管理されていたプロジェクトであり、無関係です。

Deep Furiganaは、日本語の漢字に特殊な読み方を割り当てて、中二心をくすぐるものです。
特殊な読み方（発音でなく文脈的な表現を表している）とすることが多く、外国人にとって日本語の学習の障害になっているということです。

<p align="center">
  <img width="350px" src="https://cloud.githubusercontent.com/assets/4949982/26139347/e9927c1a-3b0a-11e7-9022-1720ac82ec5a.jpg">
</p>
<div align="center"> 図1. Fateシリーズのアルテラの場面、生命と文明を（おまえたち）と呼ばせる </div>

## つまりどういうことなのか
Deep Furiganaがある文脈の前後にて、生命と、文明と、おまえたちは、同等の意味を持っていると考えられます。
意味が等しい、かつ近しい中二っぽい単語をDeepFuriganaとして適応すればよいということになりそうです

ほかの例として、「男は故郷（ルビ：テキサス）のことを考えていた」これは、Deep Furiganaですが、これが書かれている小説の作中では、この”テキサス”と”故郷”には同じような単語の周辺分布を持つはずです。
1. つまり故郷とテキサスは、似た意味として使われるのではないかという仮説が立ちます。
2. Deep Furiganaを多用するコンテンツは中二病に深く罹患したコンテンツ（アニメ・ゲーム・ラノベ等）などがアメリカの4chan掲示板で多いと報告されています

## 課題
Deep Furiganaは本来、発音する音でないですが、文脈的・意味的には、Deep Furiganaに言い換えられるということがわかりました。  

しかも、言い換えたコンテンツが中二病的な文章になっているという制約が入っていそうです。

Deep Furiganaをコンピュータに自動的に振らせることは可能なのでしょうか。いくつかの方法を使えば可能なように思わます。

1（文脈的に使用法が類似しており）かつ2(できるだけ中二っぽい単語の選択)が、最もDeep Furiganaらしいといえそうです

普通のニュースなどの文章にDeep Furiganを振ってみましょう。

機械学習でやっていきます。

## 説明と、システム全体図
やろうとしていることは、手間はおおいですが、単純です。 
1. fastTextのセンテンスベクタライザで、小説家になろうの文章と、Yahoo Newsの文章をベクトル化します。
2. 小説家になろうの文章をLabel1とし、Yahoo Newsの文章をLabel2とします
3. このラベルを当てられるようにliblinearでlogistic-regressionで学習していきます
4. ロジスティック回帰は確率として表現できるので、確率値で**分類できるモデル\*1**を構築します
5. **任意の文章の単語を(マルコフ)サンプリングで意味が近いという制約を課したまま、小説家になろうの単語に変換\*2**します
6. このサンプリングした変換候補の文章の中からもっとも小説家になろうの文章であると\*1を**騙せた文章を採択**します
7. 再帰構造になっており、**\*2に戻ります**

これをプログラムに落としたら、たくさんの前処理を含む、巨大なプロシージャになってしまいました。  
これは言われただけでは作ることが難しいシステムですが、全体の流れを正確に自分の中でイメージして、把握しておくことで、構築が容易になります。
<p align="center">
  <img width="450px" src="https://cloud.githubusercontent.com/assets/4949982/26141801/9d2686e2-3b18-11e7-9c2b-0bda1691ad0e.png">
</p>
<div align="center"> 図2. 各単語のベクトル化と、変換に使う候補である単語の一覧を作成 </div>

<p align="center">
  <img width="450px" src="https://cloud.githubusercontent.com/assets/4949982/26141811/a8e9513a-3b18-11e7-969e-12fb991328ca.png">
</p>
<div align="center"> 図3. Yahoo Newsとなろうの文章を文章全体でベクトル化して、判別問題に変換する </div>

<p align="center">
  <img width="500px" src="https://cloud.githubusercontent.com/assets/4949982/26141825/b16735d4-3b18-11e7-9348-f7c9e402eafd.png">
</p>
<div align="center"> 図4. 2,3で作成したモデルをもとに、変換候補の単語をサンプリング、なろうの確率が最大となる変換を見つける </div>


## 実験環境
データセット
```console
Yahoo News 100000記事 
小説家になろう、各ランクイン作品40位まで
```
学習ツール
```console
liblinear
fastText
```
パラメータ等
```cosole
liblinear( logistic, L2-loss )
fastText( nchargram=disable, dimentions=256, epoch=5 )
```
実行環境
```console
- Ubuntu 17.04
- Core i5
- 16GByteMemory
```

## 実験結果
実際にこのプログラムを走らせると、このようになります

## iterationごとの、なろう確率の変化
最初はあまりなろうっぽくないもととなる文章ですが、様々な単語選択をすることで、だんだん判別機を騙しに行けるようになってきます  
これは、実はGANの敵対的学習に影響を受けており、SeqGANの知識を使いまわしています[1]。  
  
<p align="center">
  <img width="400px" src="https://cloud.githubusercontent.com/assets/4949982/26144801/3303727c-3b25-11e7-9fa3-9a6b5cefc1f5.png">
</p>
<div align="center"> 図5. イテレーション（単語の探索をする）ごとに、判別機をだんだん騙せるようになっている </div>

### sample.1
基の文章が、このような感じ
```console
大きな要因の一つにツイッターやフェイスブック、ブログの普及で、他者の私生活の情報が手に入りやすくな>ったことが挙げられるのではないでしょうか。以前よりも、他者と比べる材料がずっと増えたわけです。
```
なろうに可能な限り意味を保持し続けて、単語を置き換えた場合
```console
大きな俗名の一つにツイッターやヴァイオレンス、絵日記の普及で、一片のコウウンキの情報が両手に入りやすくなったことが挙げられるのではないでしょうか。前回よりも、一片と比べる硝石がずっと増えたわけです。
```
これを連結すると
```console
大きな要因<<俗名>>の一つにツイッターやフェイスブック<<ヴァイオレンス>>、ブログ<<絵日記>>の普及で、他者<<一片>> の私生活<<コウウンキ>>の情報が手<<両手>>に入りやすくなったことが挙げられるのではないでしょうか。以前<<前回>>よりも、他者<<一片>>と比べる材料<<硝石>>がずっと増えたわけです。
```
このようになります。
ちょっと中二チックですね。
Facebookがヴァイオレンス（暴力？）と近しいとか、まぁ、メンタルに関する攻撃といって差し支えないので、いいでしょう。
ブログを絵日記という文脈で言い換えたり、以前を前回と言い換えたりすると中二属性が上がります。

### sample.2
もとの文章はこのようになっています
```console
「ランサムウエア(身代金要求型ウイルス)」という名のマルウエア(悪意を持ったソフトウエア)がインターネット上で大きな話題になっている。報道によると、5月12日以来、ランサムウエアの新種「WannaCry」の被害がすでに150カ国23万件以上に及んでおり、その被害は日に日に拡大中だ。
```
変換後がこのようになる
```console
「ランサムウエア(捕虜ヴェンデン型カイコ)」という片羽のマルブラトップ(敵意を持ったソフトウエア)が奈良公園で大きな話題になっている。命令違反によると、5月12日以来、ランサムウエアの冥界「WannaCry」の二次被害がすでに150カ国23万件半数に及んでおり、その二次被害は日に日に産地偽装オオトカゲだ。
```
これを連結すると
```console
「ランサムウエア(身代金<<捕虜>>要求<<ヴェンデン>>型ウイルス<<カイコ>>)」という名<<片羽>>のマルウエア(悪意<<敵意>>を持ったソフトウエア)がインターネット上<<奈良公園>>で大きな話題になっている。報道<<命令違反>>によると、5月12日以来、ランサムウエアの新種<<冥界>>「WannaCry」の被害<<二次被害>>がすでに150カ国23万件以上<<半数>>に及んでおり、その被害<<二次被害>>は日に日に拡大<<産地偽装>>中<<オオトカゲ>>だ。
```
悪意が敵意になったり、新種が冥界になったり、以上が半数になったり、被害が二次被害になったりします。なろうでは、こういう単語が好まれるのかもしれません  
拡大中が産地偽装オオトカゲとなったのは、作品に対する依存性があるのだと思います  

## まとめ
おお、これが、DeepFuriganaかって感動はちょっとずれた視点になりましたが、意味を維持しつつ、学習対象のラノベ等に近づけるという技も可能でした  
  
例えば、判別機をCharLevelCNNにしてみると、より人間書く小説に近い文体になるということがあるかもしれません  
  
学習データの作品に強く影響を受ける、かつ、特定の作品の特定のシーンに寄ることがあって、エッチな単語に言い換え続けてしまう方にだましに行ってしまうというのもありました
  
問題設定によってはDeepFurigana以外にも使えそうです 

## コード
```console
https://github.com/GINK03/DeepFurigana
```
前処理用のコード
```
$ python3 narou_deal.py
```

引数説明
- --step1: 小説家になろうをスクレイピング
- --step2: 小説家になろうを形態素解析
- --step3: Yahoo Newsを必要件数取得（ローカルに記事をダウンロードしている必要あり）
- --step4: なとうとYahooの学習の初期値依存性をなくすために、データを混ぜる
- --step5: fastTextで単語の分散表現を獲得
- --step6: fastTextの出力をgensimに変換する
- --step7: なろうとYahooの判別機を作るために、データにラベルを付ける
- --step8: テキスト情報をベクトル化するためのモデルを作ります
- --step9: なろうに存在する名詞を取り出して変換候補をつくります
- --step10: なろうと、Yahooのテキストデータをベクトル化します
- --step11: liblinearで学習します
- --step12: データセットが正しく動くことを確認します

DeepFuriganaを探索的に探していくプログラム
```console
$ python3 executor.py
```

## データセット
5.5GByteもあるのでDropboxは使えないし、どうしようと悩んでいたのですが、TwitterでBitTorrentのプロトコルを利用するといいみたいなお話をいただき、自宅のサーバを立ち上げっぱなしにすることで、簡単に構築できそうなので、TorrentFileで配布したいと思います。
  
日本ではまだアカデミアや企業の研究者が使えるトラッカーが無いように見えるので、いずれ、どなたかが立ち上げる必要がありますね。  
(つけっぱなしで放っておけるWindowが今無いのでしばらくお待ちください)

```console
$ {open what your using torrent-client} deep_furigana_vars.torrent
```

## 参考文献
[1] [SeqGAN論文](https://arxiv.org/pdf/1609.05473.pdf?)  
[2] [The Bittorrent P2P File-Sharing System: Measurements and Analysis](http://www.csd.uoc.gr/~hy558/papers/bittorrent.pdf)  
