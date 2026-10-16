"""Hand corrections to the scope filter: in-scope articles the maps leave out anyway.

enrich.py's kind label decides what is on the map. A review of every deified_person
and every modern-tradition article (2026-10-15) found four ways that label goes
wrong, each a rule rather than a case-by-case call:

  - not divine at all: an office, a symbol, satire, a being from a book of imaginary
    ones, or a relative of the deified rather than the deified person
  - a biography whose lead never says the person was worshipped. Sonnet supplied the
    deification from its own knowledge, often correctly, but the article is about the
    person, the text gives the role map nothing divine to place it by, and the same
    kind of article went the other way elsewhere (Julius Caesar and Alexander the
    Great are not_a_being)
  - worshipped only by their own movement, and living or dead since 2000: too recent
    for the worship to have become a tradition
  - the person behind a deity whose own article is also on the map

Keyed by article title, and load_corpus() asserts every title is still in scope, so a
re-fetch or re-enrichment that renames or reclassifies one fails loudly instead of
quietly bringing it back.
"""

NOT_DIVINE = {
    "Sapa Inca",  # the office of Inca emperor
    "Joyboy",  # sourced to Encyclopedia of Things That Never Were and One Piece
    "Flying Spaghetti Monster",  # satire
    "Baphomet",  # an occult symbol; the worship was a charge at the Templar trials
    "Aiwass",  # the voice that dictated The Book of the Law; a messenger, not worshipped
    "Lutunasobasoba",  # his godhood comes from an 1870s missionary story competition
    "Rana Vachhraj",  # brother of the deified Rana Jashraj and Harkor
    "Eve Frank",  # declared a messiah, which in Judaism is not a divinity
    "Hanitu",  # the Bunun idea of spirit in general (deity_concept)
    "Ubusunagami",  # the class of tutelary kami of one's birthplace (deity_concept)
}

LEAD_NEVER_SAYS_WORSHIPPED = {
    # Chinese
    "Confucius",
    "Koxinga",
    "Yue Fei",
    "Hua Tuo",
    "Li Bing (Qin)",
    "Wu Zixu",
    "Wu Feng (Qing dynasty)",
    "Zhang Xun (Tang dynasty)",
    "Wei Huacun",
    # Japanese
    "Tokugawa Ieyasu",
    "Toyotomi Hideyoshi",
    "Hoshina Masayuki",
    "Nogi Maresuke",
    "Yoshida Shōin",
    "Prince Sawara",
    "Kōga Saburō",
    "Konko Daijin",
    "Taira no Masakado",
    "Ōnishi Aijirō",
    # Korean
    "Ch'oe Yŏng",
    "Kim Yu-sin",
    # Greek and Roman
    "Bilistiche",
    "Cleopatra",
    "Antiochus II Theos",
    "Antiochus IV Epiphanes",
    "Apollonis of Cyzicus",
    "Amphiaraus",
    "Tenes",
    "Romulus",
    "Salonia Matidia",
    # Buddhist and Hindu teachers
    "Kūkai",
    "Songtsen Gampo",
    "Fu Dashi",
    "Machig Labdrön",
    "Mahavatar Babaji",
    "Shiv Dayal Singh",
    # Southeast Asian
    "Taksin",
    "Trưng sisters",
    "Cao Lỗ",
    "Shin Iza Gawna",
    # Other
    "Emotan",
    "Manco Cápac",
}

RECENT_MOVEMENT_FIGURES = {
    "José Luis de Jesús",
    "Amy Carlson (religious leader)",
    "Jehovah Wanyonyi",
    "Adi Da",
    "Mother Meera",
    "Zahng Gil-jah",
    "Ryuho Okawa",
    "Sathya Sai Baba",
    "Nirmala Srivastava",
    "Naga Thein Hlaing",
}

# Person article -> the deity article that stays.
DUPLICATES = {
    "Sugawara no Michizane": "Tenjin (kami)",
    "Anawrahta of Launggyet": "Shwe Nawrahta",
    "Ino (mythology)": "Leucothea",
    "Dhana Kumari Bajracharya": "Kumari (goddess)",
    "Samita Bajracharya": "Kumari (goddess)",
}

EXCLUDED = NOT_DIVINE | LEAD_NEVER_SAYS_WORSHIPPED | RECENT_MOVEMENT_FIGURES | set(DUPLICATES)
