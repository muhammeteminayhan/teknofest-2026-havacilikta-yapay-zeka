"""
ortak — TEKNOFEST 2026 Havacılıkta YZ üç görevin paylaştığı çekirdek katman.

Görevler bağlantılıdır (aynı hava videosu, aynı arayüz, kare başına tek sonuç paketi):
- ego_hareket: kameranın küresel hareketini (homografi) kestirir. Görev 1'in
  hareketli/sabit ayrımı ve Görev 2'nin VO'su bu mantığı paylaşır.
- veri: kare/modalite yardımcıları (640x512 → termal otomatik tespiti vb.).
- kamera: kamera kalibrasyon yardımcıları (Görev 2'deki çözümle uyumlu).
"""
