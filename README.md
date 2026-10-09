# 卵ケース（cage 流用版）

CyberCyclist「Egg Carry Holder V2」の cage を内側のかごとして流用し、
SanderWel のような**密閉できる外殻**で包んだ携帯卵ケース。

## 構成：3種類・4パーツ

| ファイル | 役割 | 枚数 | 外形 | 樹脂量 |
|---|---|---|---|---|
| `out/cage_unit.stl` | 内側ケージ（点対称なので上下共通） | **2** | φ55.0 × H30.84 | 4.07 cm³ |
| `out/shell_bottom.stl` | 外殻・下（おねじ） | 1 | φ61.4 × H42.54 | 8.87 cm³ |
| `out/shell_top.stl` | 外殻・上（めねじ） | 1 | φ65.2 × H32.96 | 14.45 cm³ |

**組立：最大外径 φ65.2 × 全高 66.1 mm / 樹脂合計 31.4 cm³**
（SanderWel は φ63.8・樹脂 115.9 cm³）

## 設計パラメータ（`scripts/04_build_shell.py` 冒頭）

| 変数 | 値 | 意味 |
|---|---|---|
| `CLEAR` | 1.0 | cage のたわみ代（全周クリアランス） |
| `WALL` | 1.2 | 殻の肉厚 |
| `SKWALL` | 1.6 | 蓋スカートの肉厚 |
| `MAXSLOPE` | 1.0 | dR/dz の上限＝鉛直から45°。**オーバーハング禁止の肝** |
| `PITCH` | 5.0 | ねじピッチ（1条・フランク45°・台形） |
| `TDEPTH` / `TCLR` | 1.0 / 0.30 | ねじ山深さ / はめあい隙間 |
| `NECK_H` | 9.5 | ねじ首の高さ |
| `THR_TURNS` | 1.1 | おねじの巻き数＝噛み合い **396°** |
| `GRV_TURNS` | 2.1 | めねじ溝の巻き数（後述） |

ねじ：外径 φ61.4 / 谷径 φ59.4、山高2.6・溝高4.4・残り山0.6mm。
シールは首の頂面と蓋の肩が当たる**幅1.2mmの環状面**。

## 設計上の注意（ハマったところ）

1. **cage の軸は切断正方形の中心ではない。** 真の軸は (26.3, 26.3)、リム R=25.0。
   円フィットで求めている（`scripts/02`）。
2. **元の cage は点対称ではない。** 耳が 45°側 r=29.65 / 225°側 r=31.35 と非対称。
   軸まわり180°回転コピーとの union で点対称化している。
3. **耳・タブは全部削除した**（半径27.5の円筒でカット）。本家ではロッド固定に
   使っていたが、外殻があるので不要。これで最大半径 30.5→27.5。
4. **クリアランス包絡面は頂点だけから計算してはいけない。**
   耳の平らな側面には中間頂点がなく、0.62mm 食い込む。
   `subdivide_to_size(max_edge=0.4)` でメッシュを細分化してから計算する。
5. **めねじ溝はおねじより1ピッチ分下に長く伸ばす**（`GRV_TURNS = THR_TURNS + 1.0`）。
   溝がスカート下端まで届いていないと、締め込み途中でおねじが溝の端に衝突する。
   ランアウトのテーパーも溝側には付けない。
6. **ねじ溝を r/z 両方向に一律オフセットするとフランク角が変わる**（45°→58°で
   サポートが必要になった）。45°を保ったまま隙間を取ること。
7. **booleanの結果をSTLに書くと watertight が壊れることがある。**
   大きなboxとのintersectionを最後に一回かけると直る（`clean()`）。
   ただし**最後の座標移動の後**にかけること。

## 生成手順

事前に [Egg Carry Holder V2 (Thingiverse 1185864)](https://www.thingiverse.com/thing:1185864) から
`cage.stl` をダウンロードし、`ref_CyberCyclist/files/cage.stl` に置く
（参照モデルと生成物 `out/` はリポジトリに含めていない）。

```
python scripts/01_cut_quarter.py      # cage.stl -> out/work/cage_quarter.stl (2x2の1個分を切出し)
python scripts/02_make_symmetric.py   # -> out/work/cage_unit_sym.stl        (点対称化)
python scripts/03_remove_lugs.py      # -> out/cage_unit.stl                 (耳を除去)
python scripts/04_build_shell.py      # -> out/shell_bottom.stl, out/shell_top.stl
```

検証：

```
python scripts/90_verify.py     # ねじ込み干渉 / 引抜き保持 / cage干渉 / オーバーハング
python scripts/91_overhang.py   # 接地面を除いたオーバーハング面積
```

`PREVIEW_DIR=<dir>` を設定するとプレビューPNGも出力する。
依存：`pip install trimesh manifold3d pillow`

### 現在の検証結果（全てパス）

```
ねじ込み 0->540度 (10点)      : 干渉 0.00 mm3 すべて
真上に引抜き 1.5 / 3.0mm      : 87 / 94 mm3  -> ねじが軸方向を保持
cage上下 x bottom/top (4組)   : 干渉 0.00 mm3
オーバーハング(接地面を除く)  : bottom 0.1 mm2 / top 0.0 mm2
3部品すべて watertight・単一ボディ
```

## 印刷

- **bottom**：平らな底を下にして、そのまま。サポート不要。
- **top**：**天面を下にして反転**。スカートが最終層。サポート不要。
- **cage**：リムを下に、2枚。

接地面は bottom が φ20.9mm（342mm²）、top が 348mm²。ブリム推奨。

はめあいがきつい／緩い場合は `scripts/04_build_shell.py` の `TCLR`（現在0.30）を
調整して `04` から作り直す。

## 未確認

実機でのねじのはめあいと密閉性。2026-10-04 時点で印刷待ち。

## 参照元

- `ref_CyberCyclist/` — Egg Carry Holder V2 (Thingiverse 1185864)
- `ref_SanderWel/` — 3Dプリント 卵の携帯容器 (MakerWorld)
