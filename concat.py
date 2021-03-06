

import MeCab

m = MeCab.Tagger("-Owakati")
orig = "「ランサムウエア(身代金要求型ウイルス)」という名のマルウエア(悪意を持ったソフトウエア)がインターネット上で大きな話題になっている。報道によると、5月12日以来、ランサムウエアの新種「WannaCry」の被害がすでに150カ国23万件以上に及んでおり、その被害は日に日に拡大中だ。"
conv = "「 ランサムウエア ( 捕虜 ヴェンデン 型 カイコ )」 という 片羽 の マルブラ トップ ( 敵意 を  持っ た ソフトウエア ) が 奈良公園 で 大きな 話題 に なっ て いる 。 命令違反 に よる と 、 5月12日 以来 、 ランサムウエア の 冥界 「 WannaCry 」 の 二次被害 が すでに 150 カ国 23 万 件 半数 に 及ん で おり 、 その 二次被害 は 日に日に 産地偽装 オオトカゲ だ 。"

orig = m.parse(orig).strip().split()
conv = conv.split()

ws = []
for t in zip(orig, conv):
  o, c = t
  if o != c:
    o += "<<{}>>".format(c)

  ws.append(o)

print( "".join(ws) )
