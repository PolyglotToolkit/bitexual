from .candidate_selection import make_candidates
from .models import LazyModels
from .similarity_based import get_embeddings, get_candidate_scores
from .utils import make_lines, arabic2ascii
from .visualization import plot_alignment, plot_scores
from .weight_based import get_weighted_candidates
from .alignment import select_alignments
from .datamodels import SentenceComparison
from .types import Model

AR = """
كان كوكب الأمير الصغير في منطقة الكواكب المرقومة بالأرقام التالية:325 و326 و327 و328 و329 و330 فبدأ رحلته بزيارتها لعلّه يجد فيها عملاً ينصرف إليه أو عملاً يفيده.
وكان أوّل كوكب نزله موطناً لملك، فرآه مرتدياً الأرجوان والسمور ومستويّاً على عرشٍ تبدو عليه،
بالرغم من بساطته، معالم الأُبّهة والجلال. وما رأى الملك الأمير الصغير حتّى صاح قائلاً: هذا من أبناء رعيّتي.
فقال الأمير في نفسه:
كيف عرفني وهو لم يرني من قبل!
وكان يجهل أنّ العالم في نظر الملوك هو شيء على غاية البساطة: فالناس جميعاً رعيّة الملوك.
ثمّ قال الملك: ادنّ منّي فأرى وجهكّ جليّاً. وكان معتزّاً بأنّه ملك يملك على أحد الناس.
أجال الأمير لحاظه مفتِّشاً عن مكان يجلس فيه فلم يجد ذلك أن معطف الملك الفاخر السابغ كان يشغل الكوكب بجملته فظلَّ واقفاً وكان قد تعب فتثاءب.
فقال له الملك: ليس من آداب البلاط أن تتثاءب بحضرة الملك. فأنا أنهيك عن التثاؤب.
فأجاب الأمير الصغير مرتبكاً: لا أستطيع أن أمنع نفسي منه فقد كانت رحلتي طويلة ولم أذُق نوماً.
قال: إذا كان الأمر كذلك فأنا آمرك بأن تتثاءب. إنّي لم أر أحداً يتثاءب من زمان بعيد.والتثاؤب في نظري أمر غريب نادر. فتثاءب وتثاءب أيضاً. هذا أمر منّي فأطع.
قال الأمير الصغير وقد احمرّ خجلاً: إنّ أمرك هذا يثير اضطرابي فلا أقوى على التثاؤب.
قال الملك: آمرك إذن بأن تتثاءب حيناً وتمتنع حيناً. وأخذ يتمتم ويدمدم ويبدي الكدر. ذلك لأنّ الملوك
تحرص حرصاً كثيراً على أن تُحترم هيبتهم وسلطتهم فلا يتساهلون في أمر الطاعة. وكان هذا الملك مطلق السلطان غير أنّه كان طيّب النفس فلا يأمر إلاّ بما يقرب من الصواب.
ومن أقواله التي كان يرددها: إنّي لو أمرت قائداً أن يتحوّل إلى طائر من طيور البحر وعصى القائد أمري لما كان الذنب ذنبه بل ذنبي.

وسأله الأمير الصغير بصوت ينمّ عن بعض الحياء والخجل: أيأذن لي الملك بالجلوس؟
قال الملك: إنّي آمرك بالجلوس فاجلس.
وجذب إليه بعزّة وجلال ذيلاً من ذيول معطفه السموريّ. وكان الأمير يعجب من أمر الملك ويقول في نفسه: على من يملك الملك في هذا الكوكب الصغير؟
ثمّ سأل الملك قائلاً: أستميحك العذر مولاي في سؤالك عن بعض الشؤون.
فبادر الملك فقال: إنّي آمرك بأن تسألني.
قال الأمير: على من تملك يا مولاي؟
فأجاب الملك بكل بساطة: أملك على كلّ شيء.


قال الأمير: على كلّ شيء؟
قال الملك: على كلّ شيء.
"""

FR = """
Il se trouvait dans la région des astéroi"des 325, 326, 327, 328, 329 et 330. Il commença donc par les visiter pour y chercher une occupation et pour s'instruire.
La première était habitée par un roi. le roi siégeait, habillé de pourpre et d'hermine, sur un trône très simple et cependant majesteuex.
-Ah! Voilà un sujet, s'écria le roi quand il aperçut le petit prince.
Et le petit prince se demanda:
-Comment peut-il me connaître puisqu'il ne m'a encore jamais vu!
Il ne savait pas que, pour les rois, le monde est très simplifié. Tous les hommes sont des sujets.
-Approche-toi que je te voie mieux, lui dit le roi qui était tout fier d'être roi pour quelqu'un.
Le petit prince chercha des yeux oû s'asseoir, mais la planète était toute encombrée par le magnifique manteau d'hermine. Il resta donc debout, et, comme il était fatigué, il bâilla.
-Il est contraire à l'étiquette de bâiller en présence d'un roi, lui dit le monarque. Je te l'interdis.
-Je ne peux pas m'en empêcher, répondit le petit prince tout confus. J'ai fait un long voyage et je n'ai pas dormi…
-Alors, lui dit le roi, je t'ordonne de bâiller. Je n'ai vu personne bâiller depuis des années. les bâillements sont pour moi des curiosités. Allons! bâille encore. C'est un ordre.
-Ca m'intimide… je ne peux plus… fit le petit prince tout rougissant.
-Hum! Hum! répontit le roi. Alors je… je t'ordonne tantôt de bâiller et tantôt de…
Il bredouillait un peu et paraissait vexé.
Car le roi tenait essentiellement à ce que son autorité fût respectée. Il ne tolérait pas le désobéissance. C'était un monarque absolu. Mais comme il était très bon, il donnait des ordres raisonnables.
"Si j'ordonnais, disait-il couramment, si j'ordonnais à un général de se changer en oiseau de mer, et si le général n'obéissait pas, ce ne serait pas la faute du général. Ce serait ma faute."
-Puis-je m'asseoir? s'enquit timidement le petit prince.
-Je t'ordonne de t'asseoir, lui répondit le roi, qui ramena majestueusement un pan de son manteau d'hermine.
Mais le petit prince s'étonnait. la planète était minuscule. Sur quoi le roi pouvait-il bien reigner?
-Sire, lui dit-il… je vous demande pardon de vous interroger…
-Je t'ordonne de m'interroger, se hâta de dire le roi.
-Sire… sur quoi régnez-vous?
-Sur tout, répondit le roi, avec une grande simplicité.
-Sur tout?
Le roi d'un geste discret désigna sa planète, les autres planètes et les étoiles.
-Sur tout ça? dit le petit prince.
-Sur tout ça… répondit le roi.
"""


def main() -> None:

    lines_a = make_lines(AR)
    lines_b = make_lines(FR)
    lazymodels = LazyModels()
    model = lazymodels.get_model(Model.MINILM)

    candidates = make_candidates(lines_a, lines_b)
    embeddings_a = get_embeddings(model, lines_a)
    embeddings_b = get_embeddings(model, lines_b)
    scores = get_candidate_scores(candidates, embeddings_a, embeddings_b)
    for (i, j), score in scores.items():
        print(f"{i:>2}  {j:>2}  {score}")

        # SCRATCH BELOW HERE

    alignment = select_alignments(scores, len(lines_a), len(lines_b), auto_anchors=4)
    plot_alignment(alignment)
    plot_scores(scores, len(lines_a), len(lines_b))
    plot_scores({k: v**2 for k, v in scores.items()}, len(lines_a), len(lines_b))

    weight_candidates, spans_a, spans_b = get_weighted_candidates(lines_a, lines_b)

    comparisons: list[SentenceComparison] = []
    for wc, weight in weight_candidates.items():
        i, j = wc
        comparison = SentenceComparison(
            a=lines_a[i],
            b=lines_b[j],
            idx_a=i,
            idx_b=j,
            relative_overlap=weight,
            char_weight_a=spans_a[i],
            char_weight_b=spans_b[j],
        )
        comparisons.append(comparison)

    for comp in comparisons:
        if comp.punctuation_matches and comp.punctuation_a == "?":
            print(comp.a)
            print(arabic2ascii(comp.b))
            print(comp.char_weight_ratio)
            print(comp.relative_overlap)


if __name__ == "__main__":
    main()
