import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random, math, itertools, numpy as np
from wordcloud import WordCloud, STOPWORDS

@dataclass
class Comment:
    user_id:str; content:str; created_at:datetime
@dataclass
class Post:
    post_id:str; author_id:str; content:str; created_at:datetime
    comments:list=field(default_factory=list)
    views:list=field(default_factory=list)          # list[(user_id, seen_at)]
    quoted_post_ids:list=field(default_factory=list)
@dataclass
class User:
    user_id:str; attributes:dict; connections:list=field(default_factory=list)
@dataclass
class SocialData:
    users:dict; posts:dict

def synthetic_data():
    now=datetime.now()
    users_raw=[{"user_id":"alice","attributes":{"gender":"female","age":30,"region":"NA"}},
               {"user_id":"bob","attributes":{"gender":"male","age":35,"region":"EU"}},
               {"user_id":"carol","attributes":{"gender":"female","age":29,"region":"NA"}},
               {"user_id":"dave","attributes":{"gender":"male","age":22,"region":"AS"}},
               {"user_id":"eve","attributes":{"gender":"female","age":40,"region":"EU"}},
               ]
    def make_views():
        return [{"user_id":random.choice([u["user_id"] for u in users_raw]),
                 "seen_at":now-timedelta(hours=random.randint(0,72))}
                for _ in range(random.randint(3,15))]
    posts_raw=[
        {"post_id":"p1","author_id":"alice","content":"Def a post","created_at":now-timedelta(days=3),
         "views":make_views()},
        {"post_id":"p2","author_id":"bob","content":"he!","created_at":now-timedelta(days=2),
         "views":make_views()},
        {"post_id":"p3","author_id":"alice","content":"Another post","created_at":now-timedelta(hours=60),
         "views":make_views()},
        {"post_id":"p4","author_id":"carol","content":"I agree with p1","created_at":now-timedelta(hours=36),
         "views":make_views(),"quoted_post_ids":["p1"]},
        {"post_id":"p5","author_id":"dave","content":"Check it","created_at":now-timedelta(hours=24),
         "views":make_views()},
        {"post_id":"p6","author_id":"eve","content":"Important","created_at":now-timedelta(hours=12),
         "views":make_views()},
    ]
    users={u["user_id"]:User(**u) for u in users_raw}
    posts={}
    for pr in posts_raw:
        p=Post(post_id=pr["post_id"],author_id=pr["author_id"],content=pr["content"],
               created_at=pr["created_at"])
        p.views=[(v["user_id"],v["seen_at"]) for v in pr["views"]]
        p.quoted_post_ids=pr.get("quoted_post_ids",[])
        # random comment counts
        p.comments=[Comment(user_id=random.choice(list(users.keys())),
                            content="Nice!", created_at=now) for _ in range(random.randint(0,5))]
        posts[p.post_id]=p
    return SocialData(users,posts)

data=synthetic_data()


def post_score(post:Post, mode="views", alpha=1.0, beta=1.0):
    if mode=="views":
        return len(post.views)
    if mode=="comments":
        return len(post.comments)
    if mode=="blend":
        return alpha*len(post.views)+beta*len(post.comments)
    raise ValueError("Unknown post metric")

def user_score(user:User, data:SocialData, metric="views", **kwargs):
    authored=[p for p in data.posts.values() if p.author_id==user.user_id]
    if metric=="views":
        return sum(len(p.views) for p in authored)
    if metric=="num_posts":
        return len(authored)
    if metric=="comments":
        return sum(len(p.comments) for p in authored)
    raise ValueError("Unknown user metric")

def filter_users(data:SocialData, attr_filter:dict|None):
    if not attr_filter: return list(data.users.values())
    out=[]
    for u in data.users.values():
        if all(u.attributes.get(k)==v for k,v in attr_filter.items()):
            out.append(u)
    return out

# post content filter for word cloud generation 
def get_filtered_posts(data: SocialData,
                       include_keywords:list[str]|None=None,
                       exclude_keywords:list[str]|None=None,
                       attr_filter:dict|None=None):
    include = set(k.lower() for k in (include_keywords or []))
    exclude = set(k.lower() for k in (exclude_keywords or []))

    allowed_users = {u.user_id for u in filter_users(data, attr_filter)}
    out = []
    for p in data.posts.values():
        if p.author_id not in allowed_users:
            continue
        content = p.content.lower()
        if include and not any(kw in content for kw in include):
            continue
        if exclude and any(kw in content for kw in exclude):
            continue
        out.append(p)
    return out

# wordcloud 
def generate_filtered_wordcloud(data: SocialData,
                                include_keywords:list[str]|None=None,
                                exclude_keywords:list[str]|None=None,
                                attr_filter:dict|None=None,
                                max_words:int=200,
                                width:int=800, height:int=400,
                                background_color:str="white"):
    posts = get_filtered_posts(data, include_keywords, exclude_keywords, attr_filter)
    if not posts:
        print("No matching posts for the given filters.")
        return
    text = " ".join(p.content for p in posts)
    wc = WordCloud(width=width, height=height, background_color=background_color,
                   stopwords=STOPWORDS, max_words=max_words).generate(text)
    plt.figure(figsize=(width/100, height/100))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Filtered Word Cloud")
    plt.tight_layout()
    plt.show()

def build_graph(data:SocialData, *, include_views=False, include_quotes=False,
                users_subset=None, posts_subset=None):
    G=nx.DiGraph()
    users_subset=users_subset or data.users.values()
    posts_subset=posts_subset or data.posts.values()
    # add nodes
    for u in users_subset:
        G.add_node(u.user_id, type="user", obj=u)
    for p in posts_subset:
        G.add_node(p.post_id, type="post", obj=p)
    # authored edges
    for p in posts_subset:
        if p.author_id in data.users:
            if p.author_id in G:
                G.add_edge(p.author_id, p.post_id, type="authored")
    # view edges
    if include_views:
        for p in posts_subset:
            for vuid,_ in p.views:
                if vuid in G:
                    G.add_edge(p.post_id, vuid, type="viewed")
    # quote edges
    if include_quotes:
        for p in posts_subset:
            for qid in p.quoted_post_ids:
                if qid in G:
                    G.add_edge(p.post_id,qid,type="quoted")
    return G

# mode: choose to highlight posts or users
# user_criteria: dict with optional 'attributes', 'metric', 'top_n'
def visualize_social_graph(data:SocialData,
                           mode="posts",
                           # post options
                           post_metric="views", alpha=1.0, beta=1.0,
                           # user options
                           user_criteria=None,   # {"attributes":{}, "metric":"views", "top_n":3}
                           # general
                           include_views=False, include_quotes=False,
                            layout_seed=0, figsize=(9,6)):

    if mode=="posts":
        # all posts, users subset who authored them
        posts=list(data.posts.values())
        users=[data.users[p.author_id] for p in posts if p.author_id in data.users]
        p_scores={p.post_id:post_score(p,mode=post_metric,alpha=alpha,beta=beta) for p in posts}
        u_scores={u.user_id:1 for u in users}  # uniform small
    else:  # mode == users

        attr=user_criteria.get("attributes") if user_criteria else None
        metric=user_criteria.get("metric","views") if user_criteria else "views"
        top_n=user_criteria.get("top_n") if user_criteria else None
        users=filter_users(data, attr)
        u_scores={u.user_id:user_score(u,data,metric=metric) for u in users}

        if top_n:
            top_ids=set([uid for uid,_ in sorted(u_scores.items(), key=lambda kv:kv[1], reverse=True)[:top_n]])
            users=[u for u in users if u.user_id in top_ids]
            u_scores={uid:score for uid,score in u_scores.items() if uid in top_ids}

        posts=[p for p in data.posts.values() if p.author_id in {u.user_id for u in users}]
        p_scores={p.post_id:post_score(p,mode="views") for p in posts}  # small
    G=build_graph(data, include_views=include_views, include_quotes=include_quotes,
                  users_subset=users, posts_subset=posts)
    # layout
    pos=nx.spring_layout(G, seed=layout_seed)
    sizes=[]
    colors=[]
    for n,d in G.nodes(data=True):
        if d['type']=="user":
            sizes.append(300+u_scores.get(n,1)*30)
            colors.append("#ffa64d")   # orange
        else:
            sizes.append(200+p_scores.get(n,1)*20)
            colors.append("#4682B4")   # steel blue
    # draw
    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(G,pos,arrowstyle="->",arrowsize=6,width=0.6,alpha=0.4,
                            edge_color="grey")
    nx.draw_networkx_nodes(G,pos,node_size=sizes,node_color=colors,alpha=0.9)
    nx.draw_networkx_labels(G,pos,font_size=8)
    plt.axis("off")
    
    
    # legend
    from matplotlib.lines import Line2D
    legend_elems=[Line2D([0],[0],marker='o',color='w',label='User',markerfacecolor='#ffa64d',markersize=8),
                  Line2D([0],[0],marker='o',color='w',label='Post',markerfacecolor='#4682B4',markersize=8)]
    plt.legend(handles=legend_elems,loc='upper right')
    title_prefix="Important Posts" if mode=="posts" else "Interesting Users"
    plt.title(f"{title_prefix} — metric: {post_metric if mode=='posts' else metric}")
    plt.tight_layout()
    plt.show()

def compute_trending_score(post: Post, window_hours: int = 24, decay: float = 0.1):
    """
    Trending score based on number of recent views, weighted by recency.
    More recent views have higher weight.
    """
    now = datetime.now()
    score = 0.0
    for user_id, seen_at in post.views:
        age = (now - seen_at).total_seconds() / 3600.0
        if age <= window_hours:
            weight = math.exp(-decay * age)  # exponential decay by age
            score += weight
    return score

def trending_posts_report(data: SocialData,
                          window_hours: int = 24,
                          top_n: int = 5,
                          include_keywords: list[str] | None = None,
                          exclude_keywords: list[str] | None = None,
                          attr_filter: dict | None = None,
                          decay: float = 0.1,
                          show_wordcloud: bool = True):
    """
    Generate a trending post report based on view activity.
    Filters posts by keywords and user attributes.
    """
    # Step 1: Filter posts by user attributes and keywords
    posts = get_filtered_posts(data, include_keywords, exclude_keywords, attr_filter)
    if not posts:
        print("No matching posts found.")
        return

    # Step 2: Compute trending score
    scored_posts = [(p, compute_trending_score(p, window_hours, decay)) for p in posts]
    scored_posts.sort(key=lambda x: x[1], reverse=True)
    top_posts = scored_posts[:top_n]

    print(f"\n📈 Top {top_n} Trending Posts (last {window_hours}h):\n")
    for rank, (p, score) in enumerate(top_posts, 1):
        print(f"{rank}. Post ID: {p.post_id} | Author: {p.author_id} | Views: {len(p.views)} | Score: {score:.2f}")
        print(f"   Content: {p.content}")
        print(f"   Created At: {p.created_at.strftime('%Y-%m-%d %H:%M')}\n")



# 1. Highlight important posts (blend of views+comments)
visualize_social_graph(data,
                       mode="posts",
                       post_metric="blend", alpha=1.0, beta=1.0,
                       include_views=False, layout_seed=13)

# 2. Highlight interesting users: female users with highest total views
visualize_social_graph(data,
                       mode="users",
                       user_criteria={"attributes":{"gender":"female"},
                                      "metric":"views",
                                      "top_n":None},
                       include_views=False, layout_seed=42)

# word cloud examples
# 1. NA users, posts with "post", "check" and excluding "he"
generate_filtered_wordcloud(
    data,
    include_keywords=["post", "check"],
    exclude_keywords=["he"],
    attr_filter={"region": "NA"}
)
# output: post, def

# 2. female users
generate_filtered_wordcloud(
    data,
    attr_filter={"gender": "female"}
)
# output: def, post, another, agree, important, p1

#trending posts globally
trending_posts_report(data, window_hours=48, top_n=3)

#trending posts from female users
trending_posts_report(data,
                      window_hours=48,
                      top_n=3,
                      attr_filter={"gender": "female"})

#trending posts with keywords and excluding others
trending_posts_report(data,
                      include_keywords=["post", "important"],
                      exclude_keywords=["he"],
                      attr_filter={"region": "NA"},
                      window_hours=36,
                      top_n=5)