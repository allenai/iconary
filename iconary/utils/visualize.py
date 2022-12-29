from os.path import join, dirname
from typing import List, Dict
from iconary.utils import utils
from iconary.data.datasets import IconaryGame, IconaryIcon


def get_icon_urls(data={}) -> Dict[str, str]:
  if len(data) == 0:
    src = join(dirname(dirname(__file__)), "data/icon_list.json")
    icon_data = utils.load_json_object(src)
    data.update({x["name"]: x["src"] for x in icon_data})
  return data


class GameVisualizer:
  CORRECT_WORD_STYLE = "color:green"

  def __init__(self, w=500, h=400, crop=True, crop_pad=30):
    self.w = w
    self.h = h
    self.crop = crop
    self.crop_pad = crop_pad

  def get_guesses_html(self, guesses, status, prev_status):
    all_guesses = []
    for guess, status in zip(guesses, status):
      guess_html = []
      for word, prev_stat, stat in zip(guess, prev_status, status):
        if stat == 0:
          # Incorrect
          guess_html.append(f'<span style="color:red">{word}</span>')
        elif stat != 2:
          # Partially correct
          guess_html.append(f'<span style="color:blue">{word}</span>')
        else:
          if prev_stat != 2:
            # New correct word
            guess_html.append(f'<span style="{self.CORRECT_WORD_STYLE}">{word}</span>')
          else:
            # Already guessed word
            guess_html.append(f"<span>{word}</span>")
      prev_status = status
      all_guesses.append(" ".join(guess_html))
    return all_guesses

  def render_drawing(self, drawing, container_style, kind="div"):
    return render_drawing_html(
      self.w, self.h, drawing, self.crop, self.crop_pad,
      container_style, kind=kind
    )

  def render_game(self, game: IconaryGame, phrase_html=None) -> str:
    """
    :param game: Game to render
    :param phrase_html: html to use for game phrase
    :return: HTML string that displays the game
    """
    html = [f'<div style="font-size:18px; max-width:{self.w + 50}; margin: auto">']
    prev_status = [0 if g else 2 for g in game.is_given]

    if phrase_html is None:
      phrase_html = []
      for word, stat in zip(game.game_phrase, prev_status):
        if stat == 2:
          phrase_html.append(f'<span style="{self.CORRECT_WORD_STYLE}">{word}</span>')
        else:
          phrase_html.append(f"<span>{word}</span>")

    html.append(f'<div style="text-align: center; font-size:26px; border-bottom:solid; padding-top:20px">')
    html += phrase_html
    html.append("</div>")

    container_style = "border-style:solid; margin:auto"
    html.append(f'<div>ID: {game.id}</div>')

    for state_num, state in enumerate(game.game_states):
      drawing = self.render_drawing(state.drawing, container_style)
      html.append(f'<div>Drawing {state_num+1}</div>')
      html.append('<div>')
      html += drawing
      html.append("</div>")

      guess_html = self.get_guesses_html(state.guesses, state.status, prev_status)
      if len(state.status) > 0:
        prev_status = state.status[-1]
      html.append("<ul>")
      html += [f'<li>{guess}</li>' for guess in guess_html]
      html.append("</ul>")

    html.append(f'</div>')
    return "\n".join(html)


def render_drawing_html(
    w, h, drawing: List[IconaryIcon],
    crop=False, crop_pad=0, container_style="border-style:solid",
    icon_style="", kind="div"
):
  icon_data = get_icon_urls()
  html = []
  x_offset = 0
  y_offset = 0

  if crop:
    min_x, min_y = 1, 1
    max_x, max_y = 0, 0
    for icon in drawing:
      min_x = min(min_x, icon.x - (icon.width / 2.0))
      min_y = min(min_y, icon.y - (icon.height / 2.0))
      max_x = max(max_x, icon.x + (icon.width / 2.0))
      max_y = max(max_y, icon.y + (icon.height / 2.0))

    x_offset = crop_pad - min_x * w
    y_offset = crop_pad - min_y * h
    max_w = (max_x - min_x) * w + crop_pad*2
    max_h = (max_y - min_y) * h + crop_pad*2
  else:
    max_w = w
    max_h = h

  html.append(f'<{kind} style="width:{max_w}; max-width:{max_w}; height:{max_h}; max-height:{max_h}; {container_style}">')
  for icon in drawing:
    x = icon.x * w - (icon.width * w / 2.0) + x_offset
    y = icon.y * h - (icon.height * h / 2.0) + y_offset
    style = dict(position="absolute")
    transforms = [f"translate({x}px, {y}px)"]
    if icon.rotation_degrees != 0:
      transforms.append(f"rotate({icon.rotation_degrees}deg)")
    if icon.mirrored:
      transforms.append("scaleX(-1)")

    if transforms:
      style["transform"] = " ".join(transforms)

    if icon.name == "bbox":
      style["border"] = "1px solid"
      style["width"] = str(icon.width * w) + "px"
      style["height"] = str(icon.height * h) + "px"
      style = "; ".join(k + ":" + str(v) for k, v in style.items())
      if icon_style:
        style += "; " + icon_style
      properties = dict(
        style=style,
        title=f"{icon.name}",
      )
      html.append(
        "<div " + " ".join(k + "=\"" + str(v) + "\"" for k, v in properties.items()) + " /></div>")
    else:
      style = "; ".join(k + ":" + str(v) for k, v in style.items())
      if icon_style:
        style += "; " + icon_style
      properties = dict(
        src="https:" + icon_data[icon.name],
        width=icon.width*w,
        height=icon.height * h,
        style=style,
        title=f"{icon.name}"
      )
      html.append("<img " + " ".join(k+"=\""+str(v)+"\"" for k, v in properties.items()) + " />")
  html.append(f"</{kind}>")
  return html

