---
layout: default
title: Posts
permalink: /post/
---

### All posts

{% for post in site.posts %}
{% unless post.hide %}
* [{{ post.title }}]({{post.url}}) 
{% if post.working %}{% include span.html content="[Under Construction]" %}{% endif %}
{% endunless %}
{% endfor %}