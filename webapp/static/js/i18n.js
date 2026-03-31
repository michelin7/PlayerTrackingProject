const I18N = {
    ru: {
        // Nav
        nav_process: "Обработка",
        nav_results: "Результаты",
        nav_config: "Конфигурация",
        // Footer
        footer_title: "Player Scouting & Fitness — Система видеоаналитики хоккейных матчей",
        // Index — hero
        hero_title: "Player Scouting & Fitness",
        hero_subtitle: "Полуавтоматическая система видеоаналитики хоккейных матчей: индивидуальный скаутинг игроков и оценка физических кондиций без дорогостоящих датчиков",
        // Feature cards
        feat_scouting_title: "Скаутинг по видео",
        feat_scouting_desc: "Автоматическая нарезка видео по каждому игроку для индивидуального анализа действий на льду",
        feat_fitness_title: "Физические кондиции",
        feat_fitness_desc: "Пройденная дистанция, средняя и максимальная скорость (км/ч) — данные, ранее доступные только с GPS/LPS-датчиками",
        feat_nosensor_title: "Без датчиков",
        feat_nosensor_desc: "Достаточно трансляционного видео — не нужны дорогостоящие системы оптического или инерциального трекинга",
        feat_detection_title: "Детекция и трекинг",
        feat_detection_desc: "YOLO-модель с ускорением на Apple Neural Engine + ByteTrack для непрерывного отслеживания игроков",
        feat_ident_title: "Идентификация",
        feat_ident_desc: "Классификация команд по цветам формы и распознавание номеров с учётом контекста звеньев",
        feat_review_title: "Ревизия результатов",
        feat_review_desc: "Полуавтоматический режим: система размечает, человек проверяет и корректирует",
        // Buttons
        btn_upload: "Загрузить видео",
        btn_configs: "Конфигурации",
        // Pipeline
        pipeline_title: "Конвейер обработки",
        pipe1_title: "Конфигурация",
        pipe1_desc: "Составы команд, цвета формы, защитные пары и тройки нападающих",
        pipe2_title: "Детекция + Трекинг",
        pipe2_desc: "Обнаружение игроков и непрерывное отслеживание на протяжении видео",
        pipe3_title: "Идентификация",
        pipe3_desc: "Определение команды по цвету формы, распознавание номера, сопоставление с составом",
        pipe4_title: "Ревизия",
        pipe4_desc: "Проверка автоматической разметки, исправление ошибок, назначение неопознанных треков",
        pipe5_title: "Результаты",
        pipe5_desc: "TOI, дистанция, средняя/макс. скорость, индивидуальные видеонарезки, Excel-отчёт",
        // Models
        models_title: "Статус моделей",
        models_loading: "Загрузка...",
        model_det_name: "Детекция игроков",
        model_det_desc: "YOLO — обнаружение хоккеистов (CoreML-ускорение)",
        model_seg_name: "Детекция линий",
        model_seg_desc: "YOLO — сегментация разметки для калибровки",
        model_num_name: "Распознавание номеров",
        model_num_desc: "YOLO — OCR номеров на форме",
        model_trk_name: "ByteTrack трекер",
        model_trk_desc: "IoU-ассоциация, track_buffer: 25",
        model_trkf_name: "ByteTrack (fast)",
        model_trkf_desc: "Оптимизированный трекер без ReID",
    },
    en: {
        // Nav
        nav_process: "Processing",
        nav_results: "Results",
        nav_config: "Configuration",
        // Footer
        footer_title: "Player Scouting & Fitness — Ice Hockey Video Analytics System",
        // Index — hero
        hero_title: "Player Scouting & Fitness",
        hero_subtitle: "Semi-automated ice hockey video analytics: individual player scouting and fitness assessment from broadcast footage — no expensive wearable sensors required",
        // Feature cards
        feat_scouting_title: "Video Scouting",
        feat_scouting_desc: "Automatic per-player video clips for individual on-ice performance review",
        feat_fitness_title: "Fitness Metrics",
        feat_fitness_desc: "Distance covered, average and top skating speed (km/h) — data previously available only with GPS/LPS tracking systems",
        feat_nosensor_title: "No Sensors Needed",
        feat_nosensor_desc: "Works from standard broadcast video — no costly optical or inertial tracking hardware required",
        feat_detection_title: "Detection & Tracking",
        feat_detection_desc: "YOLO model accelerated on Apple Neural Engine + ByteTrack for continuous player tracking",
        feat_ident_title: "Player Identification",
        feat_ident_desc: "Team classification by jersey color and number recognition with line-context disambiguation",
        feat_review_title: "Review & Correction",
        feat_review_desc: "Semi-automated workflow: the system labels, the human reviews and corrects",
        // Buttons
        btn_upload: "Upload Video",
        btn_configs: "Configurations",
        // Pipeline
        pipeline_title: "Processing Pipeline",
        pipe1_title: "Configuration",
        pipe1_desc: "Team rosters, jersey colors, D-pairs and forward lines",
        pipe2_title: "Detection & Tracking",
        pipe2_desc: "Player detection and continuous tracking throughout the video",
        pipe3_title: "Identification",
        pipe3_desc: "Team classification by jersey color, number recognition, roster matching",
        pipe4_title: "Review",
        pipe4_desc: "Verify auto-labeling, correct misidentifications, assign unidentified tracks",
        pipe5_title: "Results",
        pipe5_desc: "TOI, distance, avg/top speed, individual player clips, Excel report",
        // Models
        models_title: "Model Status",
        models_loading: "Loading...",
        model_det_name: "Player Detection",
        model_det_desc: "YOLO — player detection (CoreML-accelerated)",
        model_seg_name: "Line Detection",
        model_seg_desc: "YOLO — rink marking segmentation for calibration",
        model_num_name: "Number Recognition",
        model_num_desc: "YOLO — jersey number OCR",
        model_trk_name: "ByteTrack Tracker",
        model_trk_desc: "IoU association, track_buffer: 25",
        model_trkf_name: "ByteTrack (fast)",
        model_trkf_desc: "Optimized tracker without ReID",
    }
};

(function() {
    const STORAGE_KEY = "psf_lang";
    let currentLang = localStorage.getItem(STORAGE_KEY) || "ru";

    function applyLang(lang) {
        currentLang = lang;
        localStorage.setItem(STORAGE_KEY, lang);
        document.documentElement.lang = lang;
        const dict = I18N[lang] || I18N.ru;
        document.querySelectorAll("[data-i18n]").forEach(el => {
            const key = el.getAttribute("data-i18n");
            if (dict[key] !== undefined) {
                el.textContent = dict[key];
            }
        });
        const btn = document.getElementById("langToggle");
        if (btn) btn.textContent = lang === "ru" ? "EN" : "RU";
    }

    document.addEventListener("DOMContentLoaded", () => {
        applyLang(currentLang);
        const btn = document.getElementById("langToggle");
        if (btn) {
            btn.addEventListener("click", () => {
                applyLang(currentLang === "ru" ? "en" : "ru");
            });
        }
    });

    window.PSF_I18N = { applyLang, getLang: () => currentLang, t: (key) => (I18N[currentLang] || I18N.ru)[key] || key };
})();
