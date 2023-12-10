from __future__ import annotations
from typing import Iterable
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import time


class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.neutral,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )


def click_js():
    return """function audioRecord() {
    var xPathRes = document.evaluate ('//*[contains(@class, "record")]', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null); 
    xPathRes.singleNodeValue.click();}"""


def action(btn):
    """Changes button text on click"""
    if btn == 'Speak': return 'Stop'
    else: return 'Speak'


def check_btn(btn):
    """Checks for correct button text before invoking transcribe()"""
    if btn != 'Speak': raise Exception('Recording...')

