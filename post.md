---
layout: default
title: Posts
permalink: /post/
---

### All posts

{% for post in site.posts %}
{% unless post.hide %}
* [{{ post.title }}]({{post.url}}) 
{% if post.working %}{% include uc.html %}{% endif %}
{% endunless %}
{% endfor %}