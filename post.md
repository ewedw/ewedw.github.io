---
layout: default
title: Posts
permalink: /post/
---

### All posts

{% for post in site.posts %}
{% unless post.hide %}
* [{{ post.title }}]({{post.url}}) {% if post.working %}<span style="color:#FF8970;">*[under construction]*</span>{% endif %}
{% endunless %}
{% endfor %}