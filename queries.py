"""The 10 information-seeking queries (two per student) used in Parts B and C.

Each query has:
  text    - natural-language question as an organiser/analyst would ask it
  where   - ChromaDB metadata filter applied in the *filtered* run (None = pure semantic)
  why     - the retrieval requirement / business significance
"""

QUERIES = [
 # ---------------- Student 1: Hitha Mathew P (2026NS12034)
 dict(qid="Q1", student="Hitha Mathew P", bits="2026NS12034",
      text="Why were attendees unhappy with the cashless wristband payments and what happened to their money?",
      where={"$and": [{"aspect": "cashless_payment"}, {"sentiment": {"$ne": "positive"}}]},
      why="Payment failures directly hit revenue and trust; sponsor (NimbusPay) SLA review needs the evidence."),
 dict(qid="Q2", student="Hitha Mathew P", bits="2026NS12034",
      text="How did heavy rain and waterlogging affect safety and the show schedule?",
      where={"$and": [{"city": "Mumbai"}, {"doc_type": {"$in": ["incident_report", "staff_debrief", "attendee_review"]}},
                      {"month": 7}]},
      why="Monsoon contingency planning for Marine Arena: combines city + month + doc-type conditions."),
 # ---------------- Student 2: Anurag Sharda (2026NS13074)
 dict(qid="Q3", student="Anurag Sharda", bits="2026NS13074",
      text="What problems did wheelchair users and elderly attendees face reaching the stages?",
      where={"aspect": "accessibility"},
      why="Accessibility compliance audit across venues; vocabulary varies (ramp, cobblestone, carried, seating)."),
 dict(qid="Q4", student="Anurag Sharda", bits="2026NS13074",
      text="Complaints that the music was so loud it hurt people's ears",
      where={"$and": [{"genre": "EDM"}, {"rating": {"$lte": 2}}, {"doc_type": "attendee_review"}]},
      why="Hearing-safety / noise-limit evidence for EDM line-ups; three-condition filter (genre, rating, doc type)."),
 # ---------------- Student 3: Mahfuz Alam (2026NS12121)
 dict(qid="Q5", student="Mahfuz Alam", bits="2026NS12121",
      text="What dangerous crowd crush happened when everyone tried to leave after the midnight countdown?",
      where={"$and": [{"festival_name": "Neon NYE Bash"}, {"aspect": "crowd_safety"}]},
      why="Post-incident safety investigation for the NYE event; needs staff logs and attendee accounts together."),
 dict(qid="Q6", student="Mahfuz Alam", bits="2026NS12121",
      text="Was the VIP upgrade worth the money according to VIP ticket holders?",
      where={"$and": [{"ticket_category": "VIP"}, {"aspect": "vip_experience"}]},
      why="Pricing decision for premium tiers; restricted to the people who actually bought VIP."),
 # ---------------- Student 4: Nishar Ahamed A (2026NS12120)
 dict(qid="Q7", student="Nishar Ahamed A", bits="2026NS12120",
      text="Problems getting home after the show: shuttle delays, no cabs and taxi overcharging",
      where={"$and": [{"aspect": "transport"}, {"sentiment": "negative"}]},
      why="Mobility partner (UrbanRide) planning; last-mile issues described in many different words."),
 dict(qid="Q8", student="Nishar Ahamed A", bits="2026NS12120",
      text="Which artist sets did the crowd love most for energy and crowd interaction?",
      where={"$and": [{"aspect": "performance"}, {"rating": {"$gte": 4}}]},
      why="Artist re-booking and fee negotiation (links to Performance.fee_paid in Assignment-1)."),
 # ---------------- Student 5: Ansel Jo Paul (2026NS12112)
 dict(qid="Q9", student="Ansel Jo Paul", bits="2026NS12112",
      text="How did the cold desert nights affect attendees and what helped them stay warm?",
      where={"$and": [{"city": "Jaipur"}, {"aspect": "weather"}]},
      why="Venue-specific climate planning for Palace Lawns (rentals, bonfires, advisories)."),
 dict(qid="Q10", student="Ansel Jo Paul", bits="2026NS12112",
      text="What did people say about food quality, queue times and prices at the food court?",
      where={"$and": [{"festival_name": "Rhythm Nation Tech Beats"}, {"aspect": "food_beverage"},
                      {"event_date": {"$in": ["2026-09-05", "2026-09-06"]}}]},
      why="F&B vendor (GreenLeaf Foods) contract review for one festival and its two show dates."),
]

# Key facts present in the corpus for each query - used to score factual coverage of answers.
KEY_FACTS = {
    "Q1": ["nimbuspay", "refund", "debited|deducted", "top-up|top up", "riverside"],
    "Q2": ["marine arena|mumbai|starlight", "gate 3", "40 minutes", "slipp|fell", "11 july"],
    "Q3": ["wheelchair", "heritage stage|cobblestone", "ramp|uneven", "seating", "carried|volunteers"],
    "Q4": ["jungle bass|bassline", "ears|ringing", "earplug", "loud", "bass"],
    "Q5": ["neon nye|ymca", "exit|gate", "countdown|midnight|fireworks", "crush|climbed|fence", "chennai"],
    "Q6": ["vip", "8000|rs 8000", "lounge|cabana|deck", "worth", "drinks|rs 600|price"],
    "Q7": ["shuttle", "cab|taxi", "surge|rs 2500|overcharg", "urbanride|pune|jaipur|goa", "minutes|hour"],
    "Q8": ["mc rapstar|lena cross|dj tempest|groove mechanics|sufi roots|indie wanderers", "encore|singalong|sang along|jumping",
           "crowd", "set", "festival|fest|bash"],
    "Q9": ["palace lawns|jaipur|desert", "9 degrees|cold", "blanket", "bonfire", "10 pm"],
    "Q10": ["greenleaf", "45 minutes|queue", "millet|bowls|healthy", "ran out|wraps", "rhythm nation|tech beats|bengaluru"],
}
