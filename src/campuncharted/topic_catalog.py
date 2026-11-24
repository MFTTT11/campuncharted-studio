"""Curated high-CTR Shorts & Documentary topic catalog with weighted series rotation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .config import CATEGORY_EDUCATION, CATEGORY_TRAVEL_EVENTS, DEFAULT_SERIES_WEIGHTS
from .state_store import StateStore, concept_tokens, is_near_duplicate_concept, normalize_key


@dataclass(frozen=True)
class ShortTopic:
    topic_id: str
    series: str
    subject: str
    angle_hook: str
    wiki_titles: tuple[str, ...] = ()
    official_urls: tuple[str, ...] = ()
    visual_queries: tuple[str, ...] = ()
    location_required: bool = False
    location_name: str = ""
    mood: str = "ALPINE_WIND"  # ALPINE_WIND | FOREST_RAIN | CAMPFIRE_NIGHT | CANYON_BREEZE | EPIC_DISCOVERY
    category_id: str = CATEGORY_TRAVEL_EVENTS
    extra_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentaryTopic:
    slug: str
    place: str
    region: str
    wiki_title: str
    official_url: str
    commons_category: str = ""
    mood: str = "EPIC_DISCOVERY"
    visual_queries: tuple[str, ...] = ()
    hook_angle: str = ""


# =============================================================================
# 1. VIRAL SHORTS CATALOG (5 ROTATING SERIES)
# =============================================================================
SHORT_TOPICS: tuple[ShortTopic, ...] = (
    # --- SERIES 1: CAMP_MISTAKES (Survival & Campsite Rules — highest share/save potential) ---
    ShortTopic(
        topic_id="mistake_widowmaker_trees",
        series="CAMP_MISTAKES",
        subject="Dead Standing Trees (Widowmakers) Above Campsites",
        angle_hook="Why experienced campers always look UP before pitching a tent",
        wiki_titles=("Widowmaker_(forestry)",),
        visual_queries=("tall pine forest trees wind", "camping tent forest woods", "fallen tree trunk forest", "camper setting up tent forest"),
        mood="FOREST_RAIN",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("camping safety", "forest camping", "survival tips"),
    ),
    ShortTopic(
        topic_id="mistake_dry_riverbed_flash_flood",
        series="CAMP_MISTAKES",
        subject="Camping in Dry Riverbeds and Slot Canyons",
        angle_hook="Why a completely dry canyon riverbed can flood under clear blue skies",
        wiki_titles=("Flash_flood", "Slot_canyon"),
        visual_queries=("desert canyon dry riverbed", "slot canyon rock walls", "rushing river canyon water", "desert camping tent"),
        mood="CANYON_BREEZE",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("flash flood", "canyon camping", "outdoor safety"),
    ),
    ShortTopic(
        topic_id="mistake_valley_floor_cold_air",
        series="CAMP_MISTAKES",
        subject="Cold Air Drainage in Mountain Valleys at Night",
        angle_hook="Why pitching your tent at the bottom of a valley makes your night freezing cold",
        wiki_titles=("Katabatic_wind", "Inversion_(meteorology)"),
        visual_queries=("mountain valley mist sunrise", "camping tent snowy mountains", "foggy valley morning mountains", "alpine ridge camping"),
        mood="ALPINE_WIND",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("cold weather camping", "mountain camping", "winter camping"),
    ),
    ShortTopic(
        topic_id="mistake_touching_wet_tent_wall",
        series="CAMP_MISTAKES",
        subject="Why Touching a Wet Tent Wall Causes Instant Leaks",
        angle_hook="The surprising physics behind why touching your tent during rain makes it leak",
        wiki_titles=("Tent", "Surface_tension"),
        visual_queries=("rain falling on camping tent", "rainy forest camping", "water droplets tent fabric", "cozy tent inside rain"),
        mood="FOREST_RAIN",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("rain camping", "tent tips", "camping hacks"),
    ),
    ShortTopic(
        topic_id="mistake_sleeping_on_bare_ground",
        series="CAMP_MISTAKES",
        subject="Conductive Heat Loss Through Cold Ground While Camping",
        angle_hook="Why a $500 sleeping bag still freezes you without the right pad underneath",
        wiki_titles=("Sleeping_pad", "Hypothermia"),
        visual_queries=("winter camping snow tent", "camper sleeping bag tent", "frosty grass morning camp", "mountain campsite cold"),
        mood="ALPINE_WIND",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("sleeping pad", "winter camping", "camping gear"),
    ),
    ShortTopic(
        topic_id="mistake_food_scent_bear_country",
        series="CAMP_MISTAKES",
        subject="Why Toothpaste and Lip Balm Attract Wildlife to Your Tent",
        angle_hook="The one bathroom item in your backpack that brings bears straight to camp",
        wiki_titles=("Bear_cache", "American_black_bear"),
        official_urls=("https://www.nps.gov/subjects/bears/storingfood.htm",),
        visual_queries=("grizzly bear forest wilderness", "camping backpack forest trail", "campfire forest twilight", "mountain forest tent"),
        mood="CAMPFIRE_NIGHT",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("bear safety", "wilderness camping", "backpacking"),
    ),
    ShortTopic(
        topic_id="mistake_river_rocks_campfire",
        series="CAMP_MISTAKES",
        subject="Why Wet River Rocks Explode Around a Campfire",
        angle_hook="Why seasoned campers NEVER build a fire ring using smooth river stones",
        wiki_titles=("Campfire", "Steam_explosion"),
        visual_queries=("campfire flames night woods", "river stones stream water", "burning wood embers campfire", "camping night fire"),
        mood="CAMPFIRE_NIGHT",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("campfire safety", "bushcraft", "camping mistakes"),
    ),
    ShortTopic(
        topic_id="mistake_lone_tree_lightning",
        series="CAMP_MISTAKES",
        subject="Lightning Safety in Open Mountains and Lone Trees",
        angle_hook="Why hiding under the tallest tree during a storm is a deadly camping trap",
        wiki_titles=("Lightning_safety",),
        visual_queries=("lightning storm dark clouds mountain", "solitary tree meadow field", "thunderstorm rain forest", "mountain ridge storm clouds"),
        mood="FOREST_RAIN",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("lightning safety", "hiking safety", "storm camping"),
    ),
    ShortTopic(
        topic_id="mistake_drinking_clear_mountain_stream",
        series="CAMP_MISTAKES",
        subject="Why Crystal-Clear Mountain Streams Still Hide Giardia",
        angle_hook="Why the clearest glacier stream on Earth can ruin your hike in 48 hours",
        wiki_titles=("Giardiasis", "Portable_water_purification"),
        visual_queries=("crystal clear mountain stream water", "hiker drinking water river", "waterfall forest brook", "camping boiling water pot"),
        mood="ALPINE_WIND",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("water filter", "backpacking tips", "survival"),
    ),
    ShortTopic(
        topic_id="mistake_sweating_in_winter_camp",
        series="CAMP_MISTAKES",
        subject="Why Sweating Is the Hidden Enemy of Winter Camping",
        angle_hook="The counterintuitive rule winter mountaineers follow: be bold, start cold",
        wiki_titles=("Hypothermia", "Layered_clothing"),
        visual_queries=("snowy mountain hiker winter", "winter tent deep snow", "frost snowy pine forest", "cold mountain sunrise snow"),
        mood="ALPINE_WIND",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("winter hiking", "cold camping", "mountaineering"),
    ),
    ShortTopic(
        topic_id="mistake_cotton_kills_trail",
        series="CAMP_MISTAKES",
        subject="Why Outdoor Guides Say Cotton Kills in the Backcountry",
        angle_hook="Why wearing one ordinary cotton hoodie on a chilly hike can become dangerous",
        wiki_titles=("Hypothermia", "Wool"),
        visual_queries=("misty mountain trail hiker", "rainy mountain path forest", "foggy forest hiking", "camp morning coffee mountain"),
        mood="FOREST_RAIN",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("hiking gear", "backpacking mistakes", "outdoor tips"),
    ),
    ShortTopic(
        topic_id="mistake_camping_too_close_water",
        series="CAMP_MISTAKES",
        subject="The 200-Foot Rule: Why Camping Right on the Lake Edge Backfires",
        angle_hook="Why camping right beside the water brings condensation, bugs, and fines",
        wiki_titles=("Leave_No_Trace",),
        visual_queries=("lake shoreline tent sunrise", "misty alpine lake reflection", "forest lake morning fog", "mountain lake camping"),
        mood="ALPINE_WIND",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("leave no trace", "lake camping", "campsite selection"),
    ),
    ShortTopic(
        topic_id="mistake_closed_tent_condensation",
        series="CAMP_MISTAKES",
        subject="Why Sealing Your Tent Tightly Makes It Wetter Inside",
        angle_hook="Why zipping every tent vent shut in cold weather actually soaks your sleeping bag",
        wiki_titles=("Condensation", "Tent"),
        visual_queries=("camping tent morning mist", "frosty tent sunrise mountains", "camp inside tent view mountains", "alpine morning clouds"),
        mood="ALPINE_WIND",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("tent condensation", "camping hacks", "cold camping"),
    ),
    ShortTopic(
        topic_id="mistake_dead_driftwood_beach_camp",
        series="CAMP_MISTAKES",
        subject="Why High-Tide Driftwood Lines Warn Beach Campers",
        angle_hook="The beach clue that tells you if the ocean will swallow your tent at 3 AM",
        wiki_titles=("Tide", "Sneaker_wave"),
        visual_queries=("ocean waves rocky beach coast", "beach camping tent sunset", "driftwood sandy coast", "coastal cliffs sea waves"),
        mood="CANYON_BREEZE",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("beach camping", "ocean safety", "wild camping"),
    ),
    ShortTopic(
        topic_id="mistake_green_thorn_acacia_wildlife",
        series="CAMP_MISTAKES",
        subject="Why Wildlife Rangers Avoid Camping Under Fruiting & Browsing Trees",
        angle_hook="Why pitching camp under certain wild trees guarantees midnight animal visitors",
        wiki_titles=("Acacia", "Savanna"),
        visual_queries=("savanna acacia tree sunset", "african wildlife elephants trees", "safari tent wilderness", "wilderness forest twilight"),
        mood="CANYON_BREEZE",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("wildlife safety", "safari camping", "nature rules"),
    ),

    # --- SERIES 2: WILD_STAYS (Extraordinary Real Places — High-Curiosity Global Landmarks) ---
    ShortTopic(
        topic_id="wild_svalbard_polar_bear_law",
        series="WILD_STAYS",
        subject="Svalbard, Norway — The Arctic Town Where Leaving Without Bear Protection Is Illegal",
        angle_hook="The Arctic town in Norway where stepping outside town without protection breaks the law",
        wiki_titles=("Svalbard", "Longyearbyen"),
        visual_queries=("svalbard arctic snow mountains", "polar bear snow arctic", "arctic winter glacier norway", "northern lights snowy cabin"),
        location_required=False,
        location_name="Svalbard, Norway",
        mood="ALPINE_WIND",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("svalbard", "norway", "arctic"),
    ),
    ShortTopic(
        topic_id="wild_kawah_ijen_blue_fire",
        series="WILD_STAYS",
        subject="Kawah Ijen, Indonesia — The Volcano That Burns Electric Blue at Night",
        angle_hook="Why this Indonesian volcano glows electric neon blue after dark",
        wiki_titles=("Ijen",),
        visual_queries=("volcano crater night lava", "indonesia volcano smoke mountain", "blue flame fire night", "volcanic crater lake sunrise"),
        location_name="Kawah Ijen, Indonesia",
        mood="EPIC_DISCOVERY",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("kawah ijen", "volcano", "indonesia"),
    ),
    ShortTopic(
        topic_id="wild_mount_roraima_lost_world",
        series="WILD_STAYS",
        subject="Mount Roraima, South America — The Flat-Top Cloud Island Above the Rainforest",
        angle_hook="The real-life floating mountain plateau where species exist nowhere else on Earth",
        wiki_titles=("Mount_Roraima",),
        visual_queries=("table mountain clouds cliff", "misty rainforest plateau waterfall", "dramatic mountain cliff fog", "hikers mountain summit clouds"),
        location_name="Mount Roraima, South America",
        mood="EPIC_DISCOVERY",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("mount roraima", "south america", "tepui"),
    ),
    ShortTopic(
        topic_id="wild_waitomo_glowworm_caves",
        series="WILD_STAYS",
        subject="Waitomo Caves, New Zealand — The Underground Galaxy of Living Bioluminescent Lights",
        angle_hook="Why the ceiling of this New Zealand cave looks like a starry night sky",
        wiki_titles=("Waitomo_Glowworm_Caves", "Arachnocampa_luminosa"),
        visual_queries=("dark cave underground river", "glowing blue lights night cave", "new zealand green forest river", "bioluminescent glow nature"),
        location_name="Waitomo Caves, New Zealand",
        mood="EPIC_DISCOVERY",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("waitomo", "new zealand", "caves"),
    ),
    ShortTopic(
        topic_id="wild_salar_de_uyuni_mirror",
        series="WILD_STAYS",
        subject="Salar de Uyuni, Bolivia — The World's Largest Natural Sky Mirror",
        angle_hook="When rain touches this Bolivian desert, the horizon completely disappears",
        wiki_titles=("Salar_de_Uyuni",),
        visual_queries=("salt flat mirror reflection sky", "bolivia desert reflection clouds", "white salt desert sunrise", "starry night reflection water"),
        location_name="Salar de Uyuni, Bolivia",
        mood="CANYON_BREEZE",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("salar de uyuni", "bolivia", "salt flats"),
    ),
    ShortTopic(
        topic_id="wild_wadi_rum_mars_camp",
        series="WILD_STAYS",
        subject="Wadi Rum, Jordan — Sleeping Under Zero-Light-Pollution Stars on Red Martian Sand",
        angle_hook="Why filmmakers use this Jordanian desert campsite as a stand-in for Mars",
        wiki_titles=("Wadi_Rum",),
        visual_queries=("wadi rum red desert cliffs", "desert camp tent night stars", "red sand dunes canyon sunset", "milky way desert night sky"),
        location_name="Wadi Rum, Jordan",
        mood="CANYON_BREEZE",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("wadi rum", "jordan", "desert camping"),
    ),
    ShortTopic(
        topic_id="wild_danakil_depression_ethiopia",
        series="WILD_STAYS",
        subject="Danakil Depression, Ethiopia — Neon Acid Pools in One of Earth's Hottest Places",
        angle_hook="The neon yellow landscape on Earth where temperatures regularly top 50 degrees",
        wiki_titles=("Danakil_Depression", "Dallol_(hydrothermal_system)"),
        visual_queries=("volcanic sulfur yellow hot spring", "extreme desert heat salt crust", "geothermal steam crater earth", "lava crater glowing desert"),
        location_name="Danakil Depression, Ethiopia",
        mood="CANYON_BREEZE",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("danakil", "ethiopia", "extreme earth"),
    ),
    ShortTopic(
        topic_id="wild_socotra_dragon_blood",
        series="WILD_STAYS",
        subject="Socotra Island, Yemen — The Alien Forest of Umbrella Trees",
        angle_hook="Why over a third of the plant life on this remote island exists nowhere else",
        wiki_titles=("Socotra", "Dracaena_cinnabari"),
        visual_queries=("strange ancient trees desert island", "remote turquoise beach rocky coast", "desert island mountain sunset", "exotic tree canopy sky"),
        location_name="Socotra Island, Yemen",
        mood="EPIC_DISCOVERY",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("socotra", "alien landscapes", "remote island"),
    ),
    ShortTopic(
        topic_id="wild_aogashima_volcano_village",
        series="WILD_STAYS",
        subject="Aogashima, Japan — Living Inside an Active Double-Crater Ocean Volcano",
        angle_hook="The Japanese island where fewer than 200 people live INSIDE an active volcano",
        wiki_titles=("Aogashima",),
        visual_queries=("green volcanic island ocean cliffs", "japan volcano crater mist", "pacific ocean steep green cliffs", "starry sky mountain island"),
        location_name="Aogashima, Japan",
        mood="EPIC_DISCOVERY",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("aogashima", "japan", "volcano island"),
    ),
    ShortTopic(
        topic_id="wild_zhangjiajie_pillars",
        series="WILD_STAYS",
        subject="Zhangjiajie, China — Quartz Sandstone Pillars Towering Through Mountain Clouds",
        angle_hook="How rain and frost carved 3,000 floating rock towers in China's misty forest",
        wiki_titles=("Zhangjiajie_National_Forest_Park",),
        visual_queries=("zhangjiajie mountain pillars fog", "misty steep sandstone peaks forest", "clouds moving mountain cliffs", "ancient pine mountain cliff"),
        location_name="Zhangjiajie, China",
        mood="EPIC_DISCOVERY",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("zhangjiajie", "china", "national park"),
    ),
    ShortTopic(
        topic_id="wild_lencois_maranhenses_lagoons",
        series="WILD_STAYS",
        subject="Lençóis Maranhenses, Brazil — Thousands of Freshwater Lagoons Inside Desert Dunes",
        angle_hook="Why this Brazilian sand desert fills with thousands of crystal-blue swimming lagoons",
        wiki_titles=("Lençóis_Maranhenses_National_Park",),
        visual_queries=("white sand dunes blue lagoon", "brazil desert water pools", "aerial sand dunes turquoise water", "sunset desert oasis dunes"),
        location_name="Lencois Maranhenses, Brazil",
        mood="CANYON_BREEZE",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("brazil", "sand dunes", "hidden wonders"),
    ),
    ShortTopic(
        topic_id="wild_skeleton_coast_namibia",
        series="WILD_STAYS",
        subject="Skeleton Coast, Namibia — Where Towering Desert Dunes Crash Into the Atlantic",
        angle_hook="Why sailors called this foggy Namibian coastline the land made in anger",
        wiki_titles=("Skeleton_Coast",),
        visual_queries=("namibia desert dunes ocean coast", "foggy ocean beach sand dunes", "wild coastline waves mist", "desert sunset atlantic ocean"),
        location_name="Skeleton Coast, Namibia",
        mood="CANYON_BREEZE",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("namibia", "skeleton coast", "desert ocean"),
    ),
    ShortTopic(
        topic_id="wild_kamchatka_valley_geysers",
        series="WILD_STAYS",
        subject="Kamchatka Peninsula, Russia — Land of Fire, Bears, and 160 Volcanoes",
        angle_hook="The roadless wilderness peninsula with 160 volcanoes and almost no humans",
        wiki_titles=("Kamchatka_Peninsula", "Volcanoes_of_Kamchatka"),
        visual_queries=("volcano snow peak wilderness", "geothermal geyser steam valley", "brown bear river salmon wilderness", "wild volcanic coastline mist"),
        location_name="Kamchatka, Russia",
        mood="ALPINE_WIND",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("kamchatka", "volcanoes", "wilderness"),
    ),
    ShortTopic(
        topic_id="wild_fitz_roy_patagonia_wind",
        series="WILD_STAYS",
        subject="Mount Fitz Roy, Patagonia — Why Campers Anchor Their Tents With Boulders",
        angle_hook="Why Patagonia's Roaring Forties winds can flatten an ordinary tent in seconds",
        wiki_titles=("Fitz_Roy", "Roaring_Forties"),
        visual_queries=("patagonia fitz roy mountain sunrise", "windy mountain tent camping", "glacier lake granite peaks", "hiker dramatic mountains patagonia"),
        location_name="Mount Fitz Roy, Patagonia",
        mood="ALPINE_WIND",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("patagonia", "fitz roy", "backpacking"),
    ),
    ShortTopic(
        topic_id="wild_faroe_sorvagsvatn_lake",
        series="WILD_STAYS",
        subject="Sørvágsvatn, Faroe Islands — The Optical-Illusion Lake Hanging Above the Ocean",
        angle_hook="The lake in the Faroe Islands that tricks your eyes into thinking it floats over the sea",
        wiki_titles=("Sørvágsvatn",),
        visual_queries=("faroe islands green cliffs ocean", "cliff lake waterfall sea", "misty green mountains north atlantic", "dramatic sea cliffs clouds"),
        location_name="Faroe Islands",
        mood="ALPINE_WIND",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("faroe islands", "optical illusion", "nature"),
    ),

    # --- SERIES 3: GEAR_AND_HACKS (Backcountry Science, Bushcraft & Outdoor Intelligence) ---
    ShortTopic(
        topic_id="hack_pine_cone_weather",
        series="GEAR_AND_HACKS",
        subject="How Pine Cones Predict Rain Hours Before a Storm",
        angle_hook="How a single pine cone on the forest floor predicts rain before your phone does",
        wiki_titles=("Conifer_cone",),
        visual_queries=("pine cone close up forest", "pine forest rain clouds", "forest floor moss pine needles", "raindrops conifer forest"),
        mood="FOREST_RAIN",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("nature hacks", "weather prediction", "bushcraft"),
    ),
    ShortTopic(
        topic_id="hack_high_altitude_boiling_water",
        series="GEAR_AND_HACKS",
        subject="Why Boiling Water Fails to Cook Camp Meals Properly at High Altitude",
        angle_hook="Why water boils at a lower temperature on mountains and ruins backpacker meals",
        wiki_titles=("High-altitude_cooking", "Boiling_point"),
        visual_queries=("camping stove boiling water mountain", "high alpine mountain summit", "camp coffee steaming mug mountains", "backcountry cooking camp"),
        mood="ALPINE_WIND",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("camp cooking", "mountain science", "backpacking"),
    ),
    ShortTopic(
        topic_id="hack_rule_of_threes_survival",
        series="GEAR_AND_HACKS",
        subject="The Survival Rule of Threes Every Solo Camper Memorizes",
        angle_hook="The 3-3-3 survival rule that tells you what actually kills lost hikers first",
        wiki_titles=("Rule_of_threes_(survival)",),
        visual_queries=("solo hiker foggy mountains", "shelter building forest woods", "campfire warmth night wilderness", "mountain compass map hiking"),
        mood="CAMPFIRE_NIGHT",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("survival rules", "solo camping", "hiking safety"),
    ),
    ShortTopic(
        topic_id="hack_moss_north_myth",
        series="GEAR_AND_HACKS",
        subject="Why the 'Moss Only Grows on the North Side' Navigation Myth Fails",
        angle_hook="Why trusting moss on trees to find north can get you even more lost",
        wiki_titles=("Moss", "Wayfinding"),
        visual_queries=("mossy tree trunk deep forest", "hiker using compass forest", "sunlight through dense forest trees", "green moss woodland"),
        mood="FOREST_RAIN",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("navigation", "outdoor myths", "survival tips"),
    ),
    ShortTopic(
        topic_id="hack_night_vision_red_headlamp",
        series="GEAR_AND_HACKS",
        subject="Why Experienced Campers Switch Headlamps to Red Light After Dark",
        angle_hook="Why one flash of a bright white headlamp destroys 30 minutes of night vision",
        wiki_titles=("Adaptation_(eye)", "Rhodopsin"),
        visual_queries=("camper headlamp night forest", "starry night sky camping tent", "night camp lantern mountains", "milky way forest night"),
        mood="CAMPFIRE_NIGHT",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("camping gear", "night hiking", "stargazing"),
    ),
    ShortTopic(
        topic_id="hack_hot_water_bottle_sleeping_bag",
        series="GEAR_AND_HACKS",
        subject="The Alpine Expedition Trick for Staying Warm Inside a Tent All Night",
        angle_hook="The simple boiled-water trick mountaineers use to heat their sleeping bag all night",
        wiki_titles=("Winter_camping",),
        visual_queries=("camp stove boiling water snow", "cozy glowing tent snow night", "winter mountains twilight stars", "snowy campsite morning"),
        mood="CAMPFIRE_NIGHT",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("winter camping", "camping hacks", "stay warm"),
    ),
    ShortTopic(
        topic_id="hack_tree_well_snow_hazard",
        series="GEAR_AND_HACKS",
        subject="Tree Wells: The Invisible Trap Beneath Deep Snowy Pines",
        angle_hook="Why stepping too close to a snow-covered evergreen tree can swallow a hiker whole",
        wiki_titles=("Tree_well",),
        visual_queries=("deep snow evergreen pine forest", "winter hiking snowshoes forest", "heavy snow pine branches", "snowy alpine woods"),
        mood="ALPINE_WIND",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("winter safety", "snow hiking", "backcountry"),
    ),
    ShortTopic(
        topic_id="hack_birch_bark_wet_firestarter",
        series="GEAR_AND_HACKS",
        subject="Why White Birch Bark Burns Even When Soaking Wet",
        angle_hook="The wild forest bark packed with natural waterproof oil that lights in pouring rain",
        wiki_titles=("Birch_bark", "Betulin"),
        visual_queries=("white birch trees forest", "starting campfire flint sparks", "rainy woodland forest trees", "warm campfire flames woods"),
        mood="CAMPFIRE_NIGHT",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("bushcraft", "fire starting", "survival skills"),
    ),
    ShortTopic(
        topic_id="hack_flash_to_bang_storm_distance",
        series="GEAR_AND_HACKS",
        subject="How to Measure Exactly How Far Away a Mountain Storm Really Is",
        angle_hook="The 5-second rule that tells you when a mountain storm is already too close",
        wiki_titles=("Thunder", "Speed_of_sound"),
        visual_queries=("lightning flash dark sky mountains", "storm clouds approaching valley", "rainy mountains dark clouds", "camping tent mountain storm"),
        mood="FOREST_RAIN",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("weather safety", "mountain hiking", "storm tips"),
    ),
    ShortTopic(
        topic_id="hack_bortle_scale_dark_skies",
        series="GEAR_AND_HACKS",
        subject="What a True Bortle Class 1 Dark-Sky Campsite Actually Looks Like",
        angle_hook="Why 80 percent of people have never seen the Milky Way from a true Class 1 dark sky",
        wiki_titles=("Bortle_scale", "Dark-sky_preserve"),
        visual_queries=("milky way night sky tent", "stars timelapse desert camp", "night sky mountains galaxy", "stargazing silhouette wilderness"),
        mood="CAMPFIRE_NIGHT",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("stargazing", "dark sky", "night camping"),
    ),

    # --- SERIES 4: EXTREME_WEATHER_NATURE (Jaw-Dropping Natural Phenomena) ---
    ShortTopic(
        topic_id="nature_crown_shyness_forest",
        series="EXTREME_WEATHER_NATURE",
        subject="Crown Shyness: Why Giant Forest Trees Refuse to Touch Each Other",
        angle_hook="Look up in certain forests and you'll see giant trees drawing rivers of sky between them",
        wiki_titles=("Crown_shyness",),
        visual_queries=("forest canopy looking up sky", "tall trees canopy sunlight", "dense forest treetops wind", "woodland trees sky pattern"),
        mood="FOREST_RAIN",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("crown shyness", "forest mysteries", "nature facts"),
    ),
    ShortTopic(
        topic_id="nature_lenticular_clouds_mountains",
        series="EXTREME_WEATHER_NATURE",
        subject="Lenticular Clouds: The UFO-Shaped Clouds That Warn Climbers of Brutal Winds",
        angle_hook="Why mountaineers turn back the moment they spot this UFO-shaped cloud on a summit",
        wiki_titles=("Lenticular_cloud",),
        visual_queries=("lenticular cloud mountain peak", "dramatic clouds snowy summit", "windy alpine peak clouds", "mount fuji cloud cap"),
        mood="ALPINE_WIND",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("lenticular clouds", "mountaineering", "weather"),
    ),
    ShortTopic(
        topic_id="nature_bioluminescent_bays",
        series="EXTREME_WEATHER_NATURE",
        subject="Bioluminescent Waves: Why Ocean Water Glows Electric Blue at Night",
        angle_hook="Why paddling through this dark coastline makes every splash explode in neon blue",
        wiki_titles=("Bioluminescence", "Dinoflagellate"),
        visual_queries=("glowing blue waves night beach", "kayak night ocean water", "coastal night waves stars", "turquoise ocean bay night"),
        mood="EPIC_DISCOVERY",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("bioluminescence", "ocean wonders", "night nature"),
    ),
    ShortTopic(
        topic_id="nature_st_elmos_fire_summit",
        series="EXTREME_WEATHER_NATURE",
        subject="St. Elmo's Fire: When Mountain Air Charges So Hard That Rocks Hum and Glow",
        angle_hook="If your hair suddenly floats on a mountain ridge, you have seconds to move",
        wiki_titles=("St._Elmo's_fire",),
        visual_queries=("mountain ridge storm clouds", "electric lightning mountain night", "dramatic alpine summit wind", "dark thunderclouds ridge"),
        mood="ALPINE_WIND",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("lightning", "mountain danger", "nature phenomena"),
    ),
    ShortTopic(
        topic_id="nature_singing_lake_ice",
        series="EXTREME_WEATHER_NATURE",
        subject="Why Frozen Mountain Lakes Make Sci-Fi Laser Sounds",
        angle_hook="The eerie sci-fi laser sound frozen mountain lakes make beneath your boots",
        wiki_titles=("Lake_ice",),
        visual_queries=("frozen mountain lake clear ice", "cracked ice frozen lake winter", "lake baikal frozen ice bubbles", "winter alpine lake snow"),
        mood="ALPINE_WIND",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("frozen lake", "winter nature", "ice sounds"),
    ),
    ShortTopic(
        topic_id="nature_alpenglow_mountains",
        series="EXTREME_WEATHER_NATURE",
        subject="Alpenglow: Why Mountain Peaks Ignite Crimson Even After Sunset",
        angle_hook="Why high mountain peaks suddenly glow bright pink minutes after the sun is gone",
        wiki_titles=("Alpenglow",),
        visual_queries=("pink mountain peak sunset alpenglow", "dolomites glowing red sunset", "alpine mountain ridge golden hour", "snowy summit purple dusk"),
        mood="ALPINE_WIND",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("alpenglow", "mountain sunrise", "nature photography"),
    ),
    ShortTopic(
        topic_id="nature_petrichor_rain_smell",
        series="EXTREME_WEATHER_NATURE",
        subject="Petrichor: Why Humans Can Smell Rain Coming Better Than Sharks Smell Blood",
        angle_hook="You can smell forest rain coming at 5 parts per trillion—here is why",
        wiki_titles=("Petrichor", "Geosmin"),
        visual_queries=("rain falling lush green forest", "wet forest soil raindrops", "misty rainforest trees rain", "rain on green leaves macro"),
        mood="FOREST_RAIN",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("petrichor", "forest rain", "nature science"),
    ),
    ShortTopic(
        topic_id="nature_glacier_blue_ice",
        series="EXTREME_WEATHER_NATURE",
        subject="Why Ancient Glacier Ice Glows Deep Electric Blue",
        angle_hook="Why thousand-year-old glacier ice turns deep electric blue instead of white",
        wiki_titles=("Blue_ice_(glacial)", "Glacier"),
        visual_queries=("blue glacier ice cave", "iceberg deep blue glacier water", "glacier crevasse mountains", "iceland blue glacier hike"),
        mood="ALPINE_WIND",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("glaciers", "blue ice", "earth science"),
    ),

    # --- SERIES 5: TURKEY_AND_EURASIA_HIDDEN (Global-Grade Spotlight on Turkey & Eurasia) ---
    ShortTopic(
        topic_id="eurasia_nemrut_crater_lake",
        series="TURKEY_AND_EURASIA_HIDDEN",
        subject="Mount Nemrut Caldera, Bitlis, Turkey — Cold and Hot Lakes Inside One Volcano",
        angle_hook="The Turkish volcano caldera holding both an icy lake and a boiling geothermal lake",
        wiki_titles=("Nemrut_(volcano)",),
        visual_queries=("volcanic crater lake mountains", "mountain lake sunrise mist turkey", "volcanic caldera green water", "highland camping lake mountains"),
        location_name="Mount Nemrut Caldera, Turkey",
        mood="EPIC_DISCOVERY",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("turkey", "nemrut caldera", "volcano lake"),
    ),
    ShortTopic(
        topic_id="eurasia_borcka_karagol_autumn",
        series="TURKEY_AND_EURASIA_HIDDEN",
        subject="Borçka Karagöl, Artvin, Turkey — The Landslide-Born Mirror Lake in the Cloud Forest",
        angle_hook="How a massive 19th-century landslide created Turkey's hidden mirror lake",
        wiki_titles=("Borçka_Karagöl_Nature_Park",),
        visual_queries=("autumn forest mirror lake reflection", "misty mountain lake pine trees", "wooden pier foggy lake forest", "colorful autumn trees mountain lake"),
        location_name="Borcka Karagol, Turkey",
        mood="FOREST_RAIN",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("artvin", "turkey nature", "hidden lakes"),
    ),
    ShortTopic(
        topic_id="eurasia_svaneti_towers_georgia",
        series="TURKEY_AND_EURASIA_HIDDEN",
        subject="Svaneti, Georgia — Medieval Stone Towers Beneath 5,000-Meter Caucasus Glaciers",
        angle_hook="The highest continuously inhabited villages in Europe guarded by medieval stone towers",
        wiki_titles=("Svaneti", "Ushguli"),
        visual_queries=("caucasus mountains georgia snow peaks", "stone tower alpine village mountains", "mountain valley hiking georgia", "glacier river valley mountains"),
        location_name="Svaneti, Georgia",
        mood="ALPINE_WIND",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("svaneti", "georgia", "caucasus"),
    ),
    ShortTopic(
        topic_id="eurasia_saklikent_gorge_turkey",
        series="TURKEY_AND_EURASIA_HIDDEN",
        subject="Saklıkent Gorge, Turkey — Walking Inside a 300-Meter-Deep Marble Canyon",
        angle_hook="Why you can only walk deep inside Turkey's 300-meter marble canyon during summer",
        wiki_titles=("Saklıkent_National_Park",),
        visual_queries=("narrow rocky river canyon gorge", "turquoise canyon river steep cliffs", "hiking through canyon stream water", "mediterranean mountain gorge"),
        location_name="Saklikent Gorge, Turkey",
        mood="CANYON_BREEZE",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("saklikent", "turkey", "canyon"),
    ),
    ShortTopic(
        topic_id="eurasia_goygol_azerbaijan",
        series="TURKEY_AND_EURASIA_HIDDEN",
        subject="Göygöl Lake, Azerbaijan — Born From an Earthquake That Split a Mountain",
        angle_hook="The sapphire mountain lake in Azerbaijan created in minutes by a medieval earthquake",
        wiki_titles=("Göygöl_(lake)", "Göygöl_National_Park"),
        visual_queries=("blue alpine lake forest mountains", "caucasus mountain forest lake", "misty green valley mountains", "clear mountain lake reflection"),
        location_name="Goygol National Park, Azerbaijan",
        mood="ALPINE_WIND",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("azerbaijan", "goygol", "caucasus"),
    ),
    ShortTopic(
        topic_id="eurasia_lycian_way_turkey",
        series="TURKEY_AND_EURASIA_HIDDEN",
        subject="The Lycian Way, Turkey — 540 Kilometers of Ancient Ruins and Turquoise Cliffs",
        angle_hook="Why hikers cross 540 kilometers of cliffside trails and sunken ruins on Turkey's coast",
        wiki_titles=("Lycian_Way",),
        visual_queries=("mediterranean cliff trail turquoise sea", "coastal mountain hiking path sunset", "ancient stone ruins forest coast", "camping overlooking turquoise sea"),
        location_name="Lycian Way, Turkey",
        mood="CANYON_BREEZE",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("lycian way", "turkey hiking", "long distance trail"),
    ),
    # --- HIGH-DEMAND CAMPING LAWS, NIGHT SURVIVAL & PRO HACKS ---
    ShortTopic(
        topic_id="wild_nordic_freedom_to_roam",
        series="WILD_STAYS",
        subject="Allemansrätten — The Nordic Law That Lets You Wild Camp Almost Anywhere for Free",
        angle_hook="The European countries where pitching your tent in the wild for free is a legal right",
        wiki_titles=("Freedom_to_roam",),
        visual_queries=("nordic pine forest tent lake", "norway fjord wild camping sunset", "sweden forest lake sunrise tent", "hiker mountain meadow scandinavia"),
        mood="ALPINE_WIND",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("wild camping", "free camping", "norway sweden"),
    ),
    ShortTopic(
        topic_id="wild_us_blm_free_camping",
        series="WILD_STAYS",
        subject="Dispersed Public Land Camping — Millions of Acres of Legal Free Campsites",
        angle_hook="How savvy campers sleep in jaw-dropping American canyons and forests completely free",
        wiki_titles=("Dispersed_camping", "Bureau_of_Land_Management"),
        visual_queries=("desert canyon camper tent sunset", "utah red rocks camping night", "mountain forest dirt road campsite", "overland camp sunrise wilderness"),
        mood="CANYON_BREEZE",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("free camping", "dispersed camping", "overlanding"),
    ),
    ShortTopic(
        topic_id="mistake_burying_campfire_zombie_fire",
        series="CAMP_MISTAKES",
        subject="Why Burying Hot Campfire Coals With Dirt Causes Underground Root Fires",
        angle_hook="Why throwing dirt over your campfire to put it out can ignite a forest days later",
        wiki_titles=("Campfire", "Smouldering"),
        visual_queries=("glowing campfire embers wood night", "pouring water over campfire smoke", "smoldering forest soil fire", "pine forest camp night"),
        mood="CAMPFIRE_NIGHT",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("campfire safety", "wildfire prevention", "camping mistakes"),
    ),
    ShortTopic(
        topic_id="nature_scary_forest_scream_red_fox",
        series="EXTREME_WEATHER_NATURE",
        subject="The Terrifying Human-Like Scream Solo Campers Hear in the Woods at 2 AM",
        angle_hook="If you hear a blood-curdling scream near your tent at 2 AM, it is usually this tiny animal",
        wiki_titles=("Red_fox",),
        visual_queries=("dark misty forest night woods", "red fox forest twilight", "solo tent glowing night woods", "owl moonlit pine forest"),
        mood="CAMPFIRE_NIGHT",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("solo camping", "forest sounds", "wildlife facts"),
    ),
    ShortTopic(
        topic_id="mistake_ten_degree_rain_hypothermia",
        series="CAMP_MISTAKES",
        subject="Why Most Backcountry Hypothermia Happens at 10°C in Rain, Not -10°C in Snow",
        angle_hook="Why a mild rainy afternoon catches more hikers off guard with hypothermia than deep winter",
        wiki_titles=("Hypothermia", "Wind_chill"),
        visual_queries=("rainy misty mountain hiker", "cold wind rain forest trail", "foggy highland mountain ridge", "camping tent heavy rain"),
        mood="FOREST_RAIN",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("hypothermia", "rain hiking", "survival rules"),
    ),
    ShortTopic(
        topic_id="mistake_grizzly_vs_black_bear_response",
        series="CAMP_MISTAKES",
        subject="Black Bear vs Grizzly Bear: Why Mixing Up the Survival Rule Can Be Fatal",
        angle_hook="The 2-second shoulder clue that tells you whether to fight back or play dead with a bear",
        wiki_titles=("Grizzly_bear", "American_black_bear"),
        official_urls=("https://www.nps.gov/subjects/bears/safety.htm",),
        visual_queries=("grizzly bear mountain forest", "black bear woodland trees", "hiker wilderness forest trail", "mountain pine forest wilderness"),
        mood="ALPINE_WIND",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("bear safety", "hiking survival", "wildlife"),
    ),
    ShortTopic(
        topic_id="hack_water_bottle_headlamp_lantern",
        series="GEAR_AND_HACKS",
        subject="The 2-Second Headlamp and Water Bottle Trick That Lights Up an Entire Tent",
        angle_hook="Why experienced backpackers strap their headlamp backward around a clear water bottle",
        wiki_titles=("Headlamp_(outdoor)", "Diffuse_reflection"),
        visual_queries=("glowing tent interior night camp", "camping headlamp forest night", "cozy campsite twilight mountains", "night camping stars tent"),
        mood="CAMPFIRE_NIGHT",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("camping hacks", "backpacking gear", "tent tips"),
    ),
    ShortTopic(
        topic_id="hack_mylar_emergency_blanket_physics",
        series="GEAR_AND_HACKS",
        subject="Why Wrapping an Emergency Space Blanket Tightly Around Yourself Fails",
        angle_hook="The tiny mistake with a $3 emergency foil blanket that stops it from keeping you warm",
        wiki_titles=("Space_blanket", "Thermal_radiation"),
        visual_queries=("cold snowy mountain hiker survival", "winter mountains mist sunset", "mountain rescue wilderness snow", "camping winter cold forest"),
        mood="ALPINE_WIND",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("survival gear", "emergency blanket", "winter safety"),
    ),
    ShortTopic(
        topic_id="wild_cliff_portaledge_camping",
        series="WILD_STAYS",
        subject="Portaledge Camping — Sleeping Suspended 3,000 Feet Up a Vertical Granite Cliff",
        angle_hook="How big-wall climbers sleep peacefully in a hanging tent 3,000 feet above the valley floor",
        wiki_titles=("Portaledge", "El_Capitan"),
        visual_queries=("steep granite cliff rock climber", "yosemite granite wall sunrise", "dramatic mountain cliff clouds", "extreme climbing vertical wall"),
        mood="ALPINE_WIND",
        category_id=CATEGORY_TRAVEL_EVENTS,
        extra_tags=("cliff camping", "portaledge", "extreme camping"),
    ),
    ShortTopic(
        topic_id="hack_drying_wet_boots_camp",
        series="GEAR_AND_HACKS",
        subject="Why Putting Soaking Wet Hiking Boots Near the Campfire Ruins Them Overnight",
        angle_hook="Never dry wet hiking boots beside a campfire—use this warm-stone expedition trick instead",
        wiki_titles=("Hiking_boot",),
        visual_queries=("hiker boots muddy forest trail", "campfire glowing evening camp", "mountain stream crossing hiker", "camping tent forest morning"),
        mood="CAMPFIRE_NIGHT",
        category_id=CATEGORY_EDUCATION,
        extra_tags=("hiking boots", "camping hacks", "backpacking tips"),
    ),
)


# =============================================================================
# 2. LONG-FORM DOCUMENTARY CATALOG (6-10 MIN LANDSCAPE 16:9)
# =============================================================================
DOCUMENTARY_TOPICS: tuple[DocumentaryTopic, ...] = (
    DocumentaryTopic(
        slug="banff_national_park",
        place="Banff National Park",
        region="Alberta, Canada",
        wiki_title="Banff_National_Park",
        official_url="https://parks.canada.ca/pn-np/ab/banff",
        commons_category="Category:Videos of Banff National Park",
        mood="ALPINE_WIND",
        visual_queries=(
            "banff national park mountains lake",
            "lake louise moraine lake canada",
            "canadian rockies glacier river forest",
            "wildlife elk bear mountain forest",
            "alpine camping tent mountains sunrise",
            "snowy mountain peaks rockies aerial",
        ),
        hook_angle="How glacial rock flour turns Banff's lakes neon turquoise—and what campers must know before entering grizzly country",
    ),
    DocumentaryTopic(
        slug="yellowstone_national_park",
        place="Yellowstone National Park",
        region="Wyoming, United States",
        wiki_title="Yellowstone_National_Park",
        official_url="https://www.nps.gov/yell/index.htm",
        commons_category="Category:Videos from Yellowstone National Park",
        mood="EPIC_DISCOVERY",
        visual_queries=(
            "yellowstone national park geyser",
            "grand prismatic spring hot spring",
            "bison herd valley river wilderness",
            "yellowstone canyon waterfall forest",
            "geothermal steam pine forest sunrise",
            "american wilderness camping mountains",
        ),
        hook_angle="Inside the supervolcano that powers more than half of Earth's geysers",
    ),
    DocumentaryTopic(
        slug="zion_national_park",
        place="Zion National Park",
        region="Utah, United States",
        wiki_title="Zion_National_Park",
        official_url="https://www.nps.gov/zion/index.htm",
        mood="CANYON_BREEZE",
        visual_queries=(
            "zion national park red canyon",
            "utah red rock cliffs river",
            "slot canyon narrows hiking water",
            "desert canyon sunset sandstone",
            "starry night desert canyon camp",
            "mountain trail steep red cliffs",
        ),
        hook_angle="How a single desert river carved 2,000-foot Navajo sandstone walls—and why the Narrows can transform in minutes",
    ),
    DocumentaryTopic(
        slug="olympic_national_park",
        place="Olympic National Park",
        region="Washington, United States",
        wiki_title="Olympic_National_Park",
        official_url="https://www.nps.gov/olym/index.htm",
        commons_category="Category:Videos from Olympic National Park",
        mood="FOREST_RAIN",
        visual_queries=(
            "temperate rainforest mossy trees fog",
            "pacific northwest rocky coast sea stacks",
            "snowy alpine mountains glacier lake",
            "rainforest river waterfall green ferns",
            "misty forest trail hiker camping",
            "wild coastline waves sunset fog",
        ),
        hook_angle="Three worlds in one park: glaciated peaks, wild Pacific sea stacks, and one of the wettest rainforests in North America",
    ),
    DocumentaryTopic(
        slug="torres_del_paine",
        place="Torres del Paine National Park",
        region="Patagonia, Chile",
        wiki_title="Torres_del_Paine_National_Park",
        official_url="https://en.wikipedia.org/wiki/Torres_del_Paine_National_Park",
        mood="ALPINE_WIND",
        visual_queries=(
            "patagonia mountains granite towers",
            "glacier ice lake patagonia chile",
            "windy mountain valley turquoise lake",
            "guanaco patagonia steppe mountains",
            "backpacking trail dramatic peaks",
            "patagonia sunrise clouds mountains",
        ),
        hook_angle="At the edge of the southern Andes, where four seasons strike in a single afternoon",
    ),
    DocumentaryTopic(
        slug="scottish_highlands_isle_of_skye",
        place="Isle of Skye & Scottish Highlands",
        region="Scotland, United Kingdom",
        wiki_title="Isle_of_Skye",
        official_url="https://en.wikipedia.org/wiki/Isle_of_Skye",
        mood="FOREST_RAIN",
        visual_queries=(
            "isle of skye scotland mountains",
            "scottish highlands misty valley lake",
            "dramatic green cliffs ocean scotland",
            "waterfall highland river fog",
            "wild camping tent highlands mountains",
            "rainy clouds green hills scotland",
        ),
        hook_angle="Why Scotland's Right to Roam makes the misty volcanic ridges of Skye one of Europe's greatest wild-camping frontiers",
    ),
    DocumentaryTopic(
        slug="swiss_alps_lauterbrunnen",
        place="Lauterbrunnen & The Swiss Alps",
        region="Bernese Oberland, Switzerland",
        wiki_title="Lauterbrunnen",
        official_url="https://en.wikipedia.org/wiki/Lauterbrunnen",
        mood="ALPINE_WIND",
        visual_queries=(
            "lauterbrunnen valley waterfall switzerland",
            "swiss alps snowy peaks green meadow",
            "alpine lake crystal water mountains",
            "glacier mountain ridge sunrise alps",
            "mountain village wooden chalets valley",
            "hiking trail swiss alps panorama",
        ),
        hook_angle="Inside the U-shaped glacier valley of 72 waterfalls that inspired fantasy worlds",
    ),
    DocumentaryTopic(
        slug="fiordland_milford_sound",
        place="Fiordland & Milford Sound",
        region="South Island, New Zealand",
        wiki_title="Fiordland_National_Park",
        official_url="https://en.wikipedia.org/wiki/Fiordland_National_Park",
        mood="FOREST_RAIN",
        visual_queries=(
            "milford sound new zealand fjord",
            "rainforest waterfalls steep cliffs ocean",
            "new zealand mountain lake mist",
            "dramatic fjord dark water clouds",
            "green mossy forest mountain stream",
            "sunset mountain peaks reflection water",
        ),
        hook_angle="Why seven meters of annual rain turns Fiordland's vertical cliffs into hundreds of temporary waterfalls",
    ),
    DocumentaryTopic(
        slug="kackar_mountains_turkey",
        place="Kaçkar Mountains & Black Sea Highlands",
        region="Rize & Artvin, Turkey",
        wiki_title="Kaçkar_Mountains",
        official_url="https://en.wikipedia.org/wiki/Ka%C3%A7kar_Mountains",
        mood="ALPINE_WIND",
        visual_queries=(
            "highland mountains sea of clouds",
            "alpine glacial lake rocky peaks",
            "misty green mountain valley river",
            "mountain ridge sunrise fog clouds",
            "camping tent alpine meadow mountains",
            "pine forest mountain waterfall mist",
        ),
        hook_angle="Above the Black Sea cloud deck: glacial lakes, wooden yaylas, and Turkey's wildest alpine trekking frontier",
    ),
    DocumentaryTopic(
        slug="arches_national_park",
        place="Arches National Park",
        region="Utah, United States",
        wiki_title="Arches_National_Park",
        official_url="https://www.nps.gov/arch/index.htm",
        mood="CANYON_BREEZE",
        visual_queries=(
            "arches national park sandstone arch",
            "utah red desert sunset canyon",
            "desert night sky milky way stars",
            "red rock desert cliffs golden hour",
            "canyon desert trail hiking rocks",
            "desert sunrise shadows sandstone",
        ),
        hook_angle="How underground salt beds and 65 million years of erosion sculpted over 2,000 natural stone arches",
    ),
    DocumentaryTopic(
        slug="sequoia_national_park",
        place="Sequoia National Park",
        region="California, United States",
        wiki_title="Sequoia_National_Park",
        official_url="https://www.nps.gov/seki/index.htm",
        mood="FOREST_RAIN",
        visual_queries=(
            "giant sequoia redwood trees forest",
            "sunbeams ancient giant forest trees",
            "sierra nevada mountains forest stream",
            "campfire giant pine forest evening",
            "misty tall trees forest canopy",
            "mountain granite peaks california",
        ),
        hook_angle="Why Earth's largest living trees actually NEED wildfire to survive and reproduce",
    ),
    DocumentaryTopic(
        slug="death_valley_national_park",
        place="Death Valley National Park",
        region="California, United States",
        wiki_title="Death_Valley_National_Park",
        official_url="https://www.nps.gov/deva/index.htm",
        mood="CANYON_BREEZE",
        visual_queries=(
            "death valley sand dunes desert",
            "salt flats polygon desert basin",
            "desert canyon colorful badlands",
            "starry night desert silhouettes",
            "extreme desert sunrise golden dunes",
            "mountain valley barren desert landscape",
        ),
        hook_angle="The science of sailing stones, 130-degree heat, and hidden pupfish thriving in North America's driest basin",
    ),
)


def _load_trending_short_topics(store: StateStore) -> list[ShortTopic]:
    """Load fresh unused Short ideas discovered by the daily YouTube Trend Radar."""
    from .state_store import read_json

    data = read_json(store.state_dir / "trending_topics.json", {})
    raw_list = data.get("topics") or []
    topics: list[ShortTopic] = []
    for row in raw_list:
        if not isinstance(row, dict) or not row.get("topic_id") or not row.get("subject"):
            continue
        topics.append(
            ShortTopic(
                topic_id=str(row["topic_id"]),
                series=str(row.get("series") or "CAMP_MISTAKES"),
                subject=str(row["subject"]),
                angle_hook=str(row.get("angle_hook") or row["subject"]),
                wiki_titles=tuple(str(x) for x in (row.get("wiki_titles") or []) if x),
                visual_queries=tuple(str(x) for x in (row.get("visual_queries") or []) if x),
                location_name=str(row.get("location_name") or ""),
                mood=str(row.get("mood") or "ALPINE_WIND"),
                category_id=str(row.get("category_id") or CATEGORY_TRAVEL_EVENTS),
                extra_tags=("camping", "outdoor tips", "wilderness"),
            )
        )
    return topics


def pick_next_short_topic(
    store: StateStore,
    *,
    series_override: str = "",
    topic_override: str = "",
) -> ShortTopic:
    """Choose the next unused Short topic blending live YouTube Trend Radar ideas + evergreen catalog."""
    trending_pool = _load_trending_short_topics(store)
    all_known = list(trending_pool) + list(SHORT_TOPICS)

    if topic_override:
        key = normalize_key(topic_override)
        for topic in all_known:
            if normalize_key(topic.topic_id) == key or key in normalize_key(topic.subject):
                return topic
        # Dynamic custom topic requested via workflow_dispatch
        slug = normalize_key(topic_override).replace(" ", "_")[:48] or "custom_topic"
        return ShortTopic(
            topic_id=f"custom_{slug}",
            series=series_override or "WILD_STAYS",
            subject=topic_override.strip(),
            angle_hook=f"The surprising truth about {topic_override.strip()}",
            visual_queries=(
                topic_override.strip(),
                "wilderness camping tent mountains",
                "nature forest river landscape",
                "mountain sunrise adventure",
            ),
            location_name=topic_override.strip(),
            mood="EPIC_DISCOVERY",
        )

    history = store.get_topic_history()
    used_topics = {normalize_key(row.get("topic_id", "")) for row in history}
    history_concepts = [
        concept_tokens(str(row.get("topic_id", "")), str(row.get("title", "")))
        for row in history
        if row.get("format") == "short"
    ]

    unused_trending: list[ShortTopic] = []
    seen_trend_locations: set[str] = set()
    seen_trend_concepts: list[set[str]] = []
    for t in trending_pool:
        if normalize_key(t.topic_id) in used_topics:
            continue
        if t.location_name and store.is_location_uploaded(t.location_name):
            continue
        loc_base = normalize_key(t.location_name.split(",", 1)[0]) if t.location_name else ""
        if loc_base and loc_base in seen_trend_locations:
            continue
        toks = concept_tokens(t.topic_id, t.subject, t.location_name)
        if is_near_duplicate_concept(toks, history_concepts) or is_near_duplicate_concept(toks, seen_trend_concepts):
            continue
        if loc_base:
            seen_trend_locations.add(loc_base)
        if toks:
            seen_trend_concepts.append(toks)
        unused_trending.append(t)

    unused_evergreen = [
        t
        for t in SHORT_TOPICS
        if normalize_key(t.topic_id) not in used_topics
        and (not t.location_name or not store.is_location_uploaded(t.location_name))
    ]

    # Blend 50% live viral trend radar ideas + 50% evergreen catalog (prefer trend if previous short was evergreen)
    recent_shorts = [r for r in history if r.get("format") == "short"]
    last_was_trend = bool(recent_shorts and str(recent_shorts[-1].get("topic_id", "")).startswith("trend_"))
    if unused_trending and not last_was_trend:
        candidates = unused_trending
    else:
        candidates = unused_evergreen or unused_trending

    if series_override:
        filtered = [t for t in (unused_trending + unused_evergreen) if t.series == series_override.upper()]
        if filtered:
            candidates = filtered

    if not candidates:
        # Never stall: recycle oldest concepts with fresh angles only after exhausting the full catalog
        candidates = list(SHORT_TOPICS)

    # Load dynamic series weights learned from YouTube Analytics
    insights = store.get_channel_insights()
    weights: dict[str, float] = dict(DEFAULT_SERIES_WEIGHTS)
    learned = insights.get("series_weights")
    if isinstance(learned, dict):
        for s_name, weight in learned.items():
            if s_name in weights and isinstance(weight, (int, float)) and weight > 0:
                weights[s_name] = float(weight)

    recent = store.recent_series(limit=5)
    available_by_series: dict[str, list[ShortTopic]] = {}
    for topic in candidates:
        available_by_series.setdefault(topic.series, []).append(topic)

    # Pick the series whose target weight exceeds its recent share the most, avoiding back-to-back repeats
    last_series = recent[-1] if recent else ""
    best_series = None
    best_deficit = float("-inf")
    total_w = sum(weights.get(s, 0.1) for s in available_by_series) or 1.0
    for s_name in available_by_series:
        target_share = weights.get(s_name, 0.1) / total_w
        recent_share = (recent.count(s_name) / len(recent)) if recent else 0.0
        repeat_penalty = 0.45 if s_name == last_series and len(available_by_series) > 1 else 0.0
        deficit = target_share - recent_share - repeat_penalty
        if deficit > best_deficit:
            best_deficit = deficit
            best_series = s_name

    pool = available_by_series[best_series or next(iter(available_by_series))]
    # Deterministic day+count hash avoids hammering the exact same topic on retry
    seed_str = f"{datetime.now(timezone.utc).date().isoformat()}:{len(used_topics)}"
    idx = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % len(pool)
    return pool[idx]


def pick_next_documentary_topic(
    store: StateStore,
    *,
    place_override: str = "",
) -> DocumentaryTopic:
    """Choose the next unproduced landscape documentary destination."""
    if place_override:
        key = normalize_key(place_override)
        for doc in DOCUMENTARY_TOPICS:
            if key in normalize_key(doc.place) or key == normalize_key(doc.slug):
                return doc
        slug = key.replace(" ", "_") or "custom_doc"
        return DocumentaryTopic(
            slug=slug,
            place=place_override.strip(),
            region=place_override.strip(),
            wiki_title=place_override.strip().replace(" ", "_"),
            official_url=f"https://en.wikipedia.org/wiki/{place_override.strip().replace(' ', '_')}",
            visual_queries=(
                place_override.strip(),
                f"{place_override.strip()} nature landscape",
                "mountains forest river wilderness",
                "camping tent mountain sunrise",
            ),
            hook_angle=f"Exploring the hidden natural wonders and backcountry stories of {place_override.strip()}",
        )

    for doc in DOCUMENTARY_TOPICS:
        if not store.is_documentary_done(doc.slug) and not store.is_documentary_done(doc.place):
            return doc

    # Fallback if all curated documentaries are already published
    return DOCUMENTARY_TOPICS[0]
