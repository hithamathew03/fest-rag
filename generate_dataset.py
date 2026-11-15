"""
generate_dataset.py
-------------------
Builds the FestiVox feedback corpus (>= 5,000 records) for Group-12, Assignment-2.

The corpus is grounded in the Assignment-1 Music Festival & Concert Management
System: the same 10 festivals, organisers, venues, stages, artists, sponsors and
ticket categories are reused, so every feedback record can be traced back to the
relational / document / graph models built earlier.

Three record types are produced:
  * attendee_review   - free-text review written by a ticket holder (has rating)
  * incident_report   - operations/security log entry written by festival staff
  * staff_debrief     - post-event debrief note written by stage/vendor managers

Text is composed from aspect-specific sentence banks plus festival-specific
"ground-truth storylines" (e.g. NimbusPay wristband failures at Riverside Fest,
waterlogging at Marine Arena). The storylines give the RAG system facts that can
be checked, and the generic banks provide realistic noise / distractors.

Output: data/festival_feedback.csv  (one row per record, text + metadata)
"""
import csv, random, hashlib, os, datetime as dt

random.seed(12)  # Group-12

# ---------------------------------------------------------------- Assignment-1 entities
FESTIVALS = {
 1: dict(name="EchoWave Summer Fest", city="Pune", start="2026-03-14", days=3, venue="Greenfield Grounds",
         stages=["Main Stage", "Acoustic Tent", "EDM Dome"], organizer="EchoWave Events",
         artists=[1, 7, 2, 4, 11, 5, 10, 8, 3], sponsors=["ZenCola Beverages", "Vortex Energy Drinks", "Flashpoint Audio"],
         tickets=["General", "VIP", "Early-Bird"]),
 2: dict(name="Starlight Monsoon Carnival", city="Mumbai", start="2026-07-10", days=3, venue="Marine Arena",
         stages=["Grand Stage"], organizer="Starlight Productions",
         artists=[3, 9, 6, 13], sponsors=["Pulse Wear Apparel", "CloudNet Telecom"], tickets=["General", "VIP"]),
 3: dict(name="Soundscape Desert Nights", city="Jaipur", start="2026-11-20", days=3, venue="Palace Lawns",
         stages=["Royal Stage", "Heritage Stage"], organizer="Soundscape Live",
         artists=[12, 14, 4, 11], sponsors=["ZenCola Beverages", "UrbanRide Mobility"],
         tickets=["General", "VIP", "Early-Bird"]),
 4: dict(name="Rhythm Nation Tech Beats", city="Bengaluru", start="2026-09-05", days=2, venue="Tech Park Amphitheatre",
         stages=["Tech Stage"], organizer="Rhythm Nation Inc",
         artists=[3, 15, 9], sponsors=["CloudNet Telecom", "Flashpoint Audio", "GreenLeaf Foods"], tickets=["General", "VIP"]),
 5: dict(name="Vibe Collective Riverside Fest", city="Delhi", start="2026-02-01", days=3, venue="Riverside Grounds",
         stages=["River Stage", "Indie Corner"], organizer="Vibe Collective",
         artists=[1, 6, 7, 14], sponsors=["NimbusPay Fintech"], tickets=["General", "VIP", "Early-Bird"]),
 6: dict(name="Soundscape Beach Bash", city="Goa", start="2026-12-28", days=3, venue="Sunset Beach Arena",
         stages=["Beach Stage", "Sunset Deck"], organizer="Soundscape Live",
         artists=[3, 9, 8, 13], sponsors=["Pulse Wear Apparel", "Vortex Energy Drinks"], tickets=["General", "VIP"]),
 7: dict(name="Neon NYE Bash", city="Chennai", start="2026-12-31", days=2, venue="YMCA Grounds Nandanam",
         stages=["NYE Main Stage", "Countdown Dome"], organizer="Neon Nights Booking",
         artists=[3, 9, 15, 8, 12], sponsors=["StaySphere Hotels", "SkyFly Airlines"], tickets=["General", "VIP"]),
 8: dict(name="Aura Indie Week", city="Kolkata", start="2026-04-10", days=3, venue="Central Maidan",
         stages=["Maidan Main", "Busker Lane"], organizer="Aura Live",
         artists=[7, 14, 13, 11, 5], sponsors=["GreenLeaf Foods"], tickets=["General", "Early-Bird"]),
 9: dict(name="Bassline Retreat", city="Goa", start="2026-10-15", days=3, venue="Sunset Beach Arena",
         stages=["Beach Stage", "Jungle Bass Tent"], organizer="Bassline Promoters",
         artists=[15, 9, 3], sponsors=["Vortex Energy Drinks"], tickets=["General", "VIP"]),
 10: dict(name="Crescendo Classical", city="Ahmedabad", start="2026-08-05", days=2, venue="Riverfront Amphitheatre",
          stages=["Sabarmati Stage", "Baithak Hall"], organizer="Crescendo Events",
          artists=[4, 11, 5], sponsors=["NimbusPay Fintech"], tickets=["General", "VIP"]),
}
ARTISTS = {
 1: ("The Midnight Echoes", "Rock"), 2: ("Arjun Kabra", "Pop"), 3: ("DJ Nova", "EDM"),
 4: ("Kavya Strings Quartet", "Classical"), 5: ("The Blue Note Collective", "Jazz"), 6: ("MC Rapstar", "Hip-Hop"),
 7: ("Indie Wanderers", "Indie"), 8: ("Lena Cross", "Pop"), 9: ("DJ Tempest", "EDM"),
 10: ("The Analog Sons", "Rock"), 11: ("Sufi Roots Ensemble", "Folk/Sufi"), 12: ("Rhea Verma", "Playback/Pop"),
 13: ("The Groove Mechanics", "Funk"), 14: ("Night Owl Collective", "Indie"), 15: ("Bassline Brothers", "EDM"),
}
FEST_WEIGHT = {1: 14, 2: 11, 3: 10, 4: 9, 5: 11, 6: 10, 7: 10, 8: 9, 9: 8, 10: 8}

ASPECTS = ["sound", "performance", "crowd_safety", "food_beverage", "ticketing_entry", "transport",
           "hygiene", "weather", "accessibility", "cashless_payment", "vip_experience", "connectivity", "lighting_visuals"]

# ---------------------------------------------------------------- generic sentence banks
# {stage} {artist} {gate} {mins} {price} {sponsor} {venue} {festival} {city}
POS = {
 "sound": ["The mix at {stage} was crisp; you could hear every vocal line clearly even at the back.",
           "Flashpoint-style line arrays at {stage} delivered clean, punchy low end without distortion.",
           "Sound engineers kept the levels balanced all evening, vocals never got buried under the drums.",
           "Even from the far side of {venue} the audio was clear and well-timed with the screens."],
 "performance": ["{artist} owned the stage and kept the crowd singing along for the entire set.",
                 "{artist} played an extended encore and spoke to the crowd between songs, felt very personal.",
                 "The {artist} set was tight, energetic and easily the highlight of my day.",
                 "{artist} brought out surprise collaborators and the crowd absolutely lost it."],
 "crowd_safety": ["Security staff were visible and polite, and the barricade team handed out water near the front.",
                  "Crowd flow was well managed with separate entry and exit lanes at {gate}.",
                  "Medical tents were clearly marked and volunteers responded quickly when someone fainted.",
                  "Never felt unsafe, even in the front pit during the headliner."],
 "food_beverage": ["Food stalls had plenty of variety and the queues moved in under {mins} minutes.",
                   "Free drinking water refill points were everywhere, which really helped.",
                   "Loved the regional food court, prices were fair for a festival at around Rs {price} a meal.",
                   "The {sponsor} stall gave out free drinks and kept lines short."],
 "ticketing_entry": ["Entry at {gate} took barely {mins} minutes, QR scanning was quick.",
                     "Wristband exchange was smooth and the staff explained re-entry rules clearly.",
                     "Early-Bird pricing was great value for a multi-day pass."],
 "transport": ["The shuttle from the city ran every 15 minutes and was well organised.",
               "Dedicated drop-off zone for cabs made arrival easy.",
               "Parking was well signposted and attendants guided us quickly."],
 "hygiene": ["Toilets were cleaned regularly and there were enough of them.",
             "Waste segregation bins were everywhere and the grounds stayed clean all evening.",
             "Hand-wash stations were stocked with soap and water throughout."],
 "weather": ["The weather was pleasant and the evening breeze made the long sets comfortable.",
             "Shaded seating areas helped during the afternoon heat."],
 "accessibility": ["The accessible viewing platform near {stage} had a great view and helpful volunteers.",
                   "Ramps and wheelchair-friendly paths were provided to all main areas.",
                   "Sign-language interpreters at {stage} were a thoughtful touch."],
 "cashless_payment": ["Cashless top-ups were instant and refunds came back to my account the next day.",
                      "Paying via the festival wristband was faster than carrying cash."],
 "vip_experience": ["VIP lounge had comfortable seating, clean washrooms and a short bar queue - worth the upgrade.",
                    "The VIP viewing deck at {stage} gave a perfect sightline to the stage.",
                    "Complimentary food in the VIP area was genuinely good, not just snacks."],
 "connectivity": ["Mobile network held up well, we could video call friends from the crowd.",
                  "Free festival Wi-Fi worked fine near the food court."],
 "lighting_visuals": ["The laser and LED visuals at {stage} were stunning and perfectly synced to the drops.",
                      "Lighting design for {artist} created a magical atmosphere.",
                      "Big screens made it easy to see the stage from the back."],
}
NEG = {
 "sound": ["Sound at {stage} kept cutting out for a few seconds at a time during {artist}'s set.",
           "Bass was so overpowering at {stage} that the vocals were just mush.",
           "Sound bleed from the neighbouring stage ruined the quieter songs.",
           "Audio and screens were out of sync at the back of {venue}."],
 "performance": ["{artist} started {mins} minutes late and cut the set short without explanation.",
                 "{artist} seemed low-energy and barely interacted with the crowd.",
                 "The {artist} set felt rushed, half the popular songs were skipped."],
 "crowd_safety": ["The crowd near {stage} got dangerously packed and there was nowhere to move.",
                  "Too few security staff at {gate}; people were jumping the barricades.",
                  "Exit routes were badly lit and people were pushing to get out after the last act.",
                  "Saw a few fights near the front and security took a long time to respond."],
 "food_beverage": ["Food queues were over {mins} minutes and several stalls ran out of food by 9 pm.",
                   "A bottle of water cost Rs {price}, which is outrageous.",
                   "Very few vegetarian options and everything was deep fried.",
                   "Drinking water stations were empty for most of the evening."],
 "ticketing_entry": ["Entry at {gate} took over {mins} minutes because only two scanners were working.",
                     "Bag checks were chaotic and they confiscated sealed water bottles.",
                     "My QR ticket would not scan and the help desk had no way to verify it.",
                     "Re-entry was not allowed even though it was promised on the website."],
 "transport": ["Waited {mins} minutes for a shuttle after the show ended; there were not enough buses.",
               "No cabs available after midnight and surge pricing was insane.",
               "Parking was a dusty field with no lighting and a long walk to the gate."],
 "hygiene": ["Toilets were filthy by the evening and there was no water in the portable units.",
             "Garbage was piling up near the food court and nobody was clearing it.",
             "Not enough toilets for the crowd, queues of {mins} minutes."],
 "weather": ["It was extremely hot in the afternoon and there was almost no shade.",
             "Rain turned the ground into mud and there were no covered areas."],
 "accessibility": ["Wheelchair access to {stage} was over uneven ground with no ramp.",
                   "The accessible toilet was locked for most of the day.",
                   "No seating for elderly attendees anywhere near the stage."],
 "cashless_payment": ["Wristband top-up failed three times and the money was deducted from my bank anyway.",
                      "Unused wristband balance has still not been refunded weeks later.",
                      "Card machines at the bars kept going offline so only cash worked."],
 "vip_experience": ["Paid Rs {vprice} for VIP and the lounge was overcrowded with no seats.",
                    "VIP viewing deck was so far from {stage} that General was honestly better.",
                    "VIP lane at {gate} was closed, so we queued with everyone else."],
 "connectivity": ["Zero mobile network inside {venue}, could not find my friends or pay via UPI.",
                  "Festival app crashed whenever the schedule was updated."],
 "lighting_visuals": ["Screens at {stage} were too small to see from the middle of the crowd.",
                      "Strobe lights were extremely intense with no warning for photosensitive people."],
}
INCIDENT = {
 "crowd_safety": ["Crowd density exceeded safe threshold near {stage}; barricade line pushed back and entry paused for {mins} minutes.",
                  "Two attendees treated for exhaustion at {stage} front pit; extra water distributed.",
                  "Altercation near {gate} broken up by security; one person escorted out."],
 "sound": ["FOH console at {stage} lost power for {mins} minutes; backup unit switched in.",
           "Decibel meter at {stage} recorded peaks above the permitted limit; engineer instructed to reduce levels."],
 "food_beverage": ["Vendor stall ran out of drinking water; emergency supply requested.",
                   "Food safety inspection flagged one stall for improper storage; stall closed for {mins} minutes."],
 "ticketing_entry": ["Scanner outage at {gate}; manual ticket verification for {mins} minutes, queue built up.",
                     "Counterfeit tickets detected at {gate}; holders denied entry."],
 "transport": ["Shuttle fleet below plan after the last act; queue at pickup exceeded {mins} minutes.",
               "Traffic police requested road closure near {venue} entrance due to congestion."],
 "hygiene": ["Portable toilet block near {stage} out of water; cleaning crew re-deployed.",
             "Waste collection fell behind schedule near the food court."],
 "weather": ["Weather warning received; stage operations suspended for {mins} minutes.",
             "Heat-related medical calls increased during afternoon sets."],
 "accessibility": ["Wheelchair user unable to reach {stage} accessible platform; volunteers assisted manually."],
 "cashless_payment": ["Payment gateway latency causing failed top-ups at multiple counters for {mins} minutes."],
 "connectivity": ["Mobile network congestion reported across {venue}; festival app sync failures."],
 "medical": [],
}
SEVERITY = ["low", "medium", "high", "critical"]

# ---------------------------------------------------------------- festival storylines (ground truth)
# (aspect, polarity, sentence, severity_hint)
STORY = {
 1: [("weather", "neg", "Day 2 at Greenfield Grounds was brutally hot, around 39 degrees, and the water stations ran dry by 4 pm."),
     ("food_beverage", "pos", "ZenCola free-refill stations near the Main Stage were a lifesaver in the Pune heat."),
     ("transport", "neg", "UrbanRide-style shuttles from Pune station were late; we stood in the pickup queue for over an hour after the Main Stage closed."),
     ("sound", "neg", "In the Acoustic Tent you could hear the EDM Dome bass thumping through Kavya Strings Quartet's quiet passages."),
     ("performance", "pos", "Lena Cross closed day 3 on the Main Stage with a stunning set and a full-crowd singalong."),
     ("lighting_visuals", "pos", "The EDM Dome visuals for DJ Nova were the best production at the festival."),
     ("ticketing_entry", "neg", "Early-Bird holders were sent to the same queue as General at Gate 2, which defeated the point of buying early.")],
 2: [("weather", "neg", "Heavy monsoon rain on 11 July flooded the area near Gate 3 at Marine Arena, ankle-deep water at the entrance."),
     ("crowd_safety", "neg", "Waterlogging made the floor near the Grand Stage slippery and several people fell during MC Rapstar's set."),
     ("performance", "neg", "Sets on 11 July were delayed by about 40 minutes because of rain and DJ Tempest's set was cut short."),
     ("vip_experience", "pos", "The covered VIP stand at Marine Arena kept us dry when the downpour started - only part of the venue that did."),
     ("connectivity", "neg", "CloudNet network went down during the rain, no calls or UPI payments inside Marine Arena for nearly an hour."),
     ("weather", "pos", "Pulse Wear handed out free ponchos at the gate which made dancing in the rain actually fun."),
     ("performance", "pos", "MC Rapstar's high-energy set had the whole Grand Stage crowd jumping despite the rain.")],
 3: [("weather", "neg", "Desert nights at Palace Lawns got very cold after 10 pm, nearly 9 degrees, and many people were not prepared."),
     ("weather", "pos", "The blanket rental counter and bonfire pits near the Royal Stage made the cold nights cosy."),
     ("accessibility", "neg", "The Heritage Stage path is cobblestone with steps; my father in a wheelchair could not reach it without being carried."),
     ("performance", "pos", "Sufi Roots Ensemble at the Heritage Stage under the stars was a spiritual experience."),
     ("sound", "pos", "Royal Stage acoustics against the palace walls were warm and natural, best sound of any festival this year."),
     ("transport", "neg", "Palace Lawns is about 25 km from Jaipur city and the UrbanRide shuttle only ran once an hour after 11 pm."),
     ("performance", "pos", "Rhea Verma sang her Bollywood hits on the Royal Stage and the whole lawn sang along.")],
 4: [("connectivity", "pos", "CloudNet booster towers meant full 5G inside Tech Park Amphitheatre, the app and UPI worked perfectly."),
     ("food_beverage", "pos", "GreenLeaf Foods court had healthy bowls and millet wraps, genuinely good festival food."),
     ("food_beverage", "neg", "GreenLeaf food court queues crossed 45 minutes during DJ Nova's set and they ran out of wraps by 8 pm."),
     ("lighting_visuals", "pos", "DJ Nova's synchronized laser show on the Tech Stage was spectacular."),
     ("ticketing_entry", "neg", "Bag check at the Outer Ring Road entrance was slow; queue took 50 minutes on the Saturday."),
     ("transport", "neg", "Outer Ring Road traffic after the show was gridlocked, cabs took over an hour to arrive."),
     ("performance", "pos", "DJ Tempest's closing set on 6 September was relentless and the crowd stayed till the last beat.")],
 5: [("cashless_payment", "neg", "NimbusPay cashless wristbands failed to top up for hours on day 1 at Riverside Grounds; money was debited but balance showed zero."),
     ("cashless_payment", "neg", "Still waiting for the NimbusPay refund of my unused wristband balance from Riverside Fest, support only sends automated replies."),
     ("cashless_payment", "neg", "Bars at the River Stage only accepted the NimbusPay wristband and when it went offline nobody could buy water."),
     ("weather", "neg", "Delhi smog in early February made the air heavy at Riverside Grounds; my eyes were burning by evening."),
     ("performance", "pos", "Night Owl Collective in the Indie Corner felt intimate, like a house gig with 500 people."),
     ("hygiene", "neg", "By day 2 the Riverside Grounds toilets near the River Stage were unusable and there was no water."),
     ("cashless_payment", "pos", "Once NimbusPay fixed the server on day 2, top-ups were instant.")],
 6: [("crowd_safety", "neg", "DJ Tempest's set at the Beach Stage was massively overcrowded; people were crushed near the front and the sand made it hard to stand."),
     ("vip_experience", "neg", "VIP at Soundscape Beach Bash cost Rs 8000 and the cabana area was nice but drinks were still Rs 600 each - not worth it."),
     ("vip_experience", "pos", "The Sunset Deck VIP cabanas had the best view of the sunset sets, a calm escape from the crowd."),
     ("performance", "pos", "The Groove Mechanics' sunset set on the Sunset Deck was pure joy, everyone dancing barefoot in the sand."),
     ("transport", "neg", "Local taxi union refused app cabs at Sunset Beach Arena and charged Rs 2500 for a 10 km ride."),
     ("hygiene", "neg", "Plastic cups everywhere on the beach by the end of the night; not good for a beach venue.")],
 7: [("crowd_safety", "neg", "After the midnight countdown at Neon NYE Bash everyone rushed to the single exit and it became a dangerous crush near the main gate."),
     ("crowd_safety", "neg", "Exit gates at YMCA Grounds were not opened in time after the fireworks; people climbed fences to get out."),
     ("transport", "pos", "Chennai Metro running extended hours till 2 am on New Year's night was a big help to get home."),
     ("lighting_visuals", "pos", "The countdown fireworks and drone show above the NYE Main Stage were breathtaking."),
     ("weather", "neg", "Chennai humidity made the Countdown Dome feel like a sauna, poor ventilation inside."),
     ("performance", "pos", "Lena Cross counting down to 2027 with the crowd was the best New Year's moment ever.")],
 8: [("food_beverage", "pos", "Local stalls at Aura Indie Week had phuchka and kathi rolls at real Kolkata prices, not festival prices."),
     ("weather", "neg", "A Nor'wester storm hit Central Maidan on 11 April; the Busker Lane tents were torn and the show stopped."),
     ("crowd_safety", "pos", "When the storm hit, staff evacuated Central Maidan calmly into covered areas within minutes, very well handled."),
     ("sound", "neg", "Busker Lane sound kept cutting out because of a faulty generator, Indie Wanderers had to play unplugged."),
     ("ticketing_entry", "pos", "Early-Bird pass for Aura Indie Week was very affordable for three days of music."),
     ("performance", "pos", "Indie Wanderers playing unplugged after the power cut turned into the most memorable set of the week.")],
 9: [("sound", "neg", "The Jungle Bass Tent at Bassline Retreat was painfully loud, my ears were ringing for two days."),
     ("sound", "pos", "Free earplugs handed out at the Bassline Retreat entrance were a responsible touch for a bass festival."),
     ("performance", "neg", "Because of the 10 pm sound curfew, Bassline Brothers' set was cut off mid-track and the crowd was furious."),
     ("crowd_safety", "neg", "Local residents near Sunset Beach Arena complained about noise and police came to enforce the curfew during Bassline Retreat."),
     ("hygiene", "neg", "The camping zone at Bassline Retreat had only a handful of showers for hundreds of campers."),
     ("lighting_visuals", "pos", "UV art installations in the Jungle Bass Tent looked incredible at night.")],
 10: [("performance", "pos", "Kavya Strings Quartet at the Baithak Hall was flawless; the audience held its breath between movements."),
      ("crowd_safety", "neg", "People kept using phones with flash during the Sabarmati Stage recital; ushers did nothing."),
      ("weather", "neg", "The air-conditioning in Baithak Hall failed on 6 August and the hall became unbearably stuffy during the Sufi Roots Ensemble set."),
      ("accessibility", "pos", "Riverfront Amphitheatre had reserved seating for senior citizens and step-free access to every row."),
      ("sound", "pos", "Unamplified acoustics in Baithak Hall were perfect for classical music."),
      ("vip_experience", "pos", "VIP pass at Crescendo Classical included a meet-and-greet with The Blue Note Collective - worth it.")],
}

OPENERS = ["", "", "", "Came with my college friends. ", "First festival with my kids. ", "Travelled from out of town for this. ",
           "Went solo this year. ", "Third time attending. ", "Bought the pass on launch day. ", "Came with my office group. ",
           "Attended all days. ", "Only made it for the last day. "]
CLOSERS_POS = ["", "Would definitely come back next year.", "Overall a great experience.", "Totally worth the ticket price.",
               "Organisers clearly learned from last year.", ""]
CLOSERS_NEG = ["", "Organisers need to fix this before next year.", "Really disappointing for the price.",
               "Not sure I will attend again.", "Expected much better from a festival this size.", ""]
CLOSERS_MIX = ["Mixed experience overall.", "Good music, poor management.", "Some things worked, others did not.", ""]


def fill(s, ctx):
    return s.format(**ctx)


def make_ctx(fid, f):
    aid = random.choice(f["artists"])
    return dict(stage=random.choice(f["stages"]), artist=ARTISTS[aid][0], gate=f"Gate {random.randint(1, 5)}",
                mins=random.choice([10, 15, 20, 25, 30, 40, 45, 60, 75, 90]), price=random.choice([80, 100, 120, 150, 200, 250, 350]),
                vprice=random.choice([5000, 5500, 6000, 6500, 7500, 8000]), sponsor=random.choice(f["sponsors"]), venue=f["venue"], festival=f["name"], city=f["city"]), aid


def gen_record(fid, idx):
    f = FESTIVALS[fid]
    ctx, aid = make_ctx(fid, f)
    doc_type = random.choices(["attendee_review", "incident_report", "staff_debrief"], [78, 14, 8])[0]
    day = random.randrange(f["days"])
    date = (dt.date.fromisoformat(f["start"]) + dt.timedelta(days=day)).isoformat()
    md = dict(festival_id=fid, festival_name=f["name"], city=f["city"], venue=f["venue"], stage=ctx["stage"],
              artist=ctx["artist"], genre=ARTISTS[aid][1], event_date=date, month=int(date[5:7]),
              doc_type=doc_type, organizer=f["organizer"], aspect2="none")
    use_story = random.random() < 0.45
    if doc_type == "incident_report":
        neg_story = [s for s in STORY[fid] if s[1] == "neg"]
        if use_story and neg_story:
            aspect, _, sent, = neg_story[random.randrange(len(neg_story))]
            body = f"Ops log, {date}: " + sent + " " + random.choice(["Escalated to duty manager.", "Logged for post-event review.",
                                                                    "Resolved on site.", "Follow-up action assigned to venue team.", ""])
        else:
            aspect = random.choice([a for a in INCIDENT if INCIDENT[a]])
            body = f"Ops log, {date}: " + fill(random.choice(INCIDENT[aspect]), ctx) + " " + random.choice(["Status: closed.", "Status: monitoring.", "Escalated to duty manager.", ""])
        md.update(aspect=aspect, severity=random.choices(SEVERITY, [30, 40, 22, 8])[0] if not use_story else random.choices(SEVERITY, [5, 30, 45, 20])[0],
                  rating=0, sentiment="negative", ticket_category="NA", author_role="security/ops")
        text = body.strip()
    elif doc_type == "staff_debrief":
        story = random.choice(STORY[fid])
        aspect, pol, sent = story
        text = (f"Debrief ({ctx['stage']}, {date}): " + sent + " " +
                random.choice(["Recommend changes for next edition.", "Vendor contract to be reviewed.", "Noted as a best practice to repeat.",
                               "Budget impact to be assessed by finance.", ""])).strip()
        md.update(aspect=aspect, severity="NA", rating=0, sentiment="positive" if pol == "pos" else "negative",
                  ticket_category="NA", author_role=random.choice(["stage manager", "vendor manager", "artist liaison", "volunteer lead"]))
    else:
        tcat = random.choices(f["tickets"], [70, 18, 12][:len(f["tickets"])])[0]
        if use_story:
            aspect, pol, main = random.choice(STORY[fid])
        else:
            aspect = random.choice(ASPECTS if tcat == "VIP" else [a for a in ASPECTS if a != "vip_experience"])
            pol = random.choices(["pos", "neg"], [55, 45])[0]
            main = fill(random.choice((POS if pol == "pos" else NEG)[aspect]), ctx)
        # second sentence: different aspect, possibly opposite polarity (mixed reviews)
        a2 = random.choice([a for a in ASPECTS if a not in (aspect, "vip_experience")])
        p2 = pol if random.random() < 0.6 else ("neg" if pol == "pos" else "pos")
        second = fill(random.choice((POS if p2 == "pos" else NEG)[a2]), ctx) if random.random() < 0.85 else ""
        if pol == p2 or not second:
            rating = random.choice([4, 5, 5]) if pol == "pos" else random.choice([1, 1, 2, 2])
            closer = random.choice(CLOSERS_POS if pol == "pos" else CLOSERS_NEG)
        else:
            rating = 3 if random.random() < 0.6 else (4 if pol == "pos" else 2)
            closer = random.choice(CLOSERS_MIX)
        text = (random.choice(OPENERS) + main + " " + second + " " + closer).strip()
        text = " ".join(text.split())
        md["aspect2"] = a2 if second else "none"
        md.update(aspect=aspect, severity="NA", rating=rating,
                  sentiment="positive" if rating >= 4 else ("negative" if rating <= 2 else "mixed"),
                  ticket_category=tcat, author_role="attendee")
    # keep metadata consistent with the text: if the text names a stage / artist
    # of this festival, use that as the record's stage / artist metadata
    for st in f["stages"]:
        if st in text:
            md["stage"] = st
            break
    for a in f["artists"]:
        if ARTISTS[a][0] in text:
            md["artist"], md["genre"] = ARTISTS[a]
            break
    md["doc_id"] = f"FB-{fid:02d}-{idx:05d}"
    md["text"] = text
    return md


def main(n=5200, out="data/festival_feedback.csv"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fids = list(FEST_WEIGHT)
    rows, seen, idx = [], set(), 0
    while len(rows) < n:
        fid = random.choices(fids, [FEST_WEIGHT[x] for x in fids])[0]
        idx += 1
        r = gen_record(fid, idx)
        h = hashlib.md5(r["text"].lower().encode()).hexdigest()
        if h in seen:
            continue  # drop exact duplicates so every record is unique
        seen.add(h)
        rows.append(r)
    cols = ["doc_id", "doc_type", "festival_id", "festival_name", "organizer", "city", "venue", "stage", "artist", "genre",
            "event_date", "month", "ticket_category", "rating", "sentiment", "aspect", "aspect2", "severity", "author_role", "text"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print(f"wrote {len(rows)} unique records to {out} (attempts={idx})")


if __name__ == "__main__":
    main()
