// GOREV 2 — ORB-SLAM3 monokuler, NEDENSEL poz kaydi.
//
// Yarismada gonderilen tahmin geri alinamaz. ORB-SLAM3'un kapanista yazdigi yorunge,
// sonradan gelen dongu kapama / BA duzeltmelerini GERIYE dogru uygular — o yorunge bizim
// puanimizi YANSITMAZ. Bu yuzden her karede TrackMonocular'in O AN dondurdugu poz yazilir.
//
// kullanim: mono_nedensel ORBvoc.txt ayar.yaml goruntu_dizini cikti.csv [fps]
//   goruntu_dizini: NNNNNN.jpg (kare numarasi dosya adindan; eksik kare atlanir)
// cikti: kare,durum,harita_degisti,cx,cy,cz,qw,qx,qy,qz  (Twc: kamera merkezi + yonelim)
//   durum: -1 hazir degil, 0 goruntu yok, 1 baslatilmadi, 2 OK, 3 yeni kayip, 4 kayip, 5 OK_KLT
#include <algorithm>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <System.h>

using namespace std;

int main(int argc, char **argv) {
    if (argc < 5) {
        cerr << "kullanim: mono_nedensel ORBvoc.txt ayar.yaml goruntu_dizini cikti.csv [fps]" << endl;
        return 1;
    }
    const string voc = argv[1], ayar = argv[2], dizin = argv[3], cikti = argv[4];
    const double fps = argc > 5 ? atof(argv[5]) : 7.5;

    vector<string> adlar;
    DIR *d = opendir(dizin.c_str());
    if (!d) { cerr << "dizin acilamadi: " << dizin << endl; return 1; }
    struct dirent *e;
    while ((e = readdir(d)) != nullptr) {
        string n = e->d_name;
        if (n.size() > 4 && n.substr(n.size() - 4) == ".jpg") adlar.push_back(n);
    }
    closedir(d);
    sort(adlar.begin(), adlar.end());
    cout << adlar.size() << " kare" << endl;

    ORB_SLAM3::System SLAM(voc, ayar, ORB_SLAM3::System::MONOCULAR, false);

    ofstream f(cikti);
    f << "kare,durum,harita_degisti,cx,cy,cz,qw,qx,qy,qz\n";
    f.precision(9);
    for (const auto &n : adlar) {
        const int k = stoi(n.substr(0, 6));
        cv::Mat im = cv::imread(dizin + "/" + n, cv::IMREAD_GRAYSCALE);
        if (im.empty()) continue;
        Sophus::SE3f Tcw = SLAM.TrackMonocular(im, k / fps);
        const int durum = SLAM.GetTrackingState();
        const bool mc = SLAM.MapChanged();
        Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Vector3f c = Twc.translation();
        Eigen::Quaternionf q = Twc.unit_quaternion();
        f << k << "," << durum << "," << (mc ? 1 : 0) << "," << c.x() << "," << c.y() << ","
          << c.z() << "," << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << "\n";
        if (k % 250 == 0) cout << "kare " << k << " durum " << durum << endl;
    }
    f.close();
    SLAM.Shutdown();
    cout << "bitti: " << cikti << endl;
    return 0;
}
