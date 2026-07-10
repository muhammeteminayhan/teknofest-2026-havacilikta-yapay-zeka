class DetectedObject:
    """Gorev 1 tespit nesnesi (sartname Sekil 17 semasi).

    NOT: Ornek arayuzun ozgun surumunde `motion_status` YOKTU; sartname (Sekil 17)
    ise tasit hareket durumu icin onu istiyor. Arkadasin `tekno_inference.py`
    ciktisi da uretiyor. Bu yuzden eklendi. `cls` semasi (URL vs duz string)
    sunucuya gore degisebilir; PAYLOAD_CLS_URL bayragi ile degistirilebilir.
    """

    # Ornek arayuz cls'i `classes/{id+1}/` URL'i olarak gonderiyordu (sunucu 1-endeksli).
    # Sunucu duz string beklerse ("0".."3") bunu False yap.
    PAYLOAD_CLS_URL = True

    def __init__(self, cls,
                 landing_status,
                 top_left_x: float,
                 top_left_y: float,
                 bottom_right_x: float,
                 bottom_right_y: float,
                 motion_status="-1",
                 ):
        self.cls = cls
        self.landing_status = str(landing_status)
        self.motion_status = str(motion_status)
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y

    def _cls_id(self) -> int:
        """cls'i tamsayi sinif id'sine cevir (tuple/list/str/int hepsini kabul et)."""
        c = self.cls
        if isinstance(c, (tuple, list)):
            c = c[0]
        return int(c)

    def create_payload(self, evaulation_server):
        cls_id = self._cls_id()
        if self.PAYLOAD_CLS_URL:
            cls_val = self.generate_api_url("classes/", str(cls_id + 1), evaulation_server)
        else:
            cls_val = str(cls_id)
        # DIKKAT: gercek sunucu alani `moving_status` diye bekliyor (sartname
        # Sekil 17'de `motion_status` yaziyordu ama format_probe ile dogrulandi:
        # sunucu yanitinda `moving_status` gorunuyor, `motion_status` yok sayiliyor).
        return {'cls': cls_val,
                'landing_status': str(self.landing_status),
                'moving_status': str(self.motion_status),
                'top_left_x': str(self.top_left_x),
                'top_left_y': str(self.top_left_y),
                'bottom_right_x': str(self.bottom_right_x),
                'bottom_right_y': str(self.bottom_right_y)
                }

    @staticmethod
    def generate_api_url(cls_endpoint, cls_id, evaulation_server):
        """Generates cls url for API usage."""
        base = evaulation_server if evaulation_server.endswith("/") else evaulation_server + "/"
        return base + cls_endpoint + cls_id + "/"
