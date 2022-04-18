//
// Created by Mike Smith on 2022/1/19.
//

#include <dsl/syntax.h>
#include <dsl/sugar.h>
#include <util/spec.h>

namespace luisa::render {

// from PBRT-v4: https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/spectrum.cpp
static constexpr const std::array cie_x_samples{
    // CIE X function values
    0.0001299000f, 0.0001458470f, 0.0001638021f, 0.0001840037f, 0.0002066902f,
    0.0002321000f, 0.0002607280f, 0.0002930750f, 0.0003293880f, 0.0003699140f,
    0.0004149000f, 0.0004641587f, 0.0005189860f, 0.0005818540f, 0.0006552347f,
    0.0007416000f, 0.0008450296f, 0.0009645268f, 0.001094949f, 0.001231154f,
    0.001368000f, 0.001502050f, 0.001642328f, 0.001802382f, 0.001995757f,
    0.002236000f, 0.002535385f, 0.002892603f, 0.003300829f, 0.003753236f,
    0.004243000f, 0.004762389f, 0.005330048f, 0.005978712f, 0.006741117f,
    0.007650000f, 0.008751373f, 0.01002888f, 0.01142170f, 0.01286901f,
    0.01431000f, 0.01570443f, 0.01714744f, 0.01878122f, 0.02074801f,
    0.02319000f, 0.02620736f, 0.02978248f, 0.03388092f, 0.03846824f,
    0.04351000f, 0.04899560f, 0.05502260f, 0.06171880f, 0.06921200f,
    0.07763000f, 0.08695811f, 0.09717672f, 0.1084063f, 0.1207672f,
    0.1343800f, 0.1493582f, 0.1653957f, 0.1819831f, 0.1986110f,
    0.2147700f, 0.2301868f, 0.2448797f, 0.2587773f, 0.2718079f,
    0.2839000f, 0.2949438f, 0.3048965f, 0.3137873f, 0.3216454f,
    0.3285000f, 0.3343513f, 0.3392101f, 0.3431213f, 0.3461296f,
    0.3482800f, 0.3495999f, 0.3501474f, 0.3500130f, 0.3492870f,
    0.3480600f, 0.3463733f, 0.3442624f, 0.3418088f, 0.3390941f,
    0.3362000f, 0.3331977f, 0.3300411f, 0.3266357f, 0.3228868f,
    0.3187000f, 0.3140251f, 0.3088840f, 0.3032904f, 0.2972579f,
    0.2908000f, 0.2839701f, 0.2767214f, 0.2689178f, 0.2604227f,
    0.2511000f, 0.2408475f, 0.2298512f, 0.2184072f, 0.2068115f,
    0.1953600f, 0.1842136f, 0.1733273f, 0.1626881f, 0.1522833f,
    0.1421000f, 0.1321786f, 0.1225696f, 0.1132752f, 0.1042979f,
    0.09564000f, 0.08729955f, 0.07930804f, 0.07171776f, 0.06458099f,
    0.05795001f, 0.05186211f, 0.04628152f, 0.04115088f, 0.03641283f,
    0.03201000f, 0.02791720f, 0.02414440f, 0.02068700f, 0.01754040f,
    0.01470000f, 0.01216179f, 0.009919960f, 0.007967240f, 0.006296346f,
    0.004900000f, 0.003777173f, 0.002945320f, 0.002424880f, 0.002236293f,
    0.002400000f, 0.002925520f, 0.003836560f, 0.005174840f, 0.006982080f,
    0.009300000f, 0.01214949f, 0.01553588f, 0.01947752f, 0.02399277f,
    0.02910000f, 0.03481485f, 0.04112016f, 0.04798504f, 0.05537861f,
    0.06327000f, 0.07163501f, 0.08046224f, 0.08973996f, 0.09945645f,
    0.1096000f, 0.1201674f, 0.1311145f, 0.1423679f, 0.1538542f,
    0.1655000f, 0.1772571f, 0.1891400f, 0.2011694f, 0.2133658f,
    0.2257499f, 0.2383209f, 0.2510668f, 0.2639922f, 0.2771017f,
    0.2904000f, 0.3038912f, 0.3175726f, 0.3314384f, 0.3454828f,
    0.3597000f, 0.3740839f, 0.3886396f, 0.4033784f, 0.4183115f,
    0.4334499f, 0.4487953f, 0.4643360f, 0.4800640f, 0.4959713f,
    0.5120501f, 0.5282959f, 0.5446916f, 0.5612094f, 0.5778215f,
    0.5945000f, 0.6112209f, 0.6279758f, 0.6447602f, 0.6615697f,
    0.6784000f, 0.6952392f, 0.7120586f, 0.7288284f, 0.7455188f,
    0.7621000f, 0.7785432f, 0.7948256f, 0.8109264f, 0.8268248f,
    0.8425000f, 0.8579325f, 0.8730816f, 0.8878944f, 0.9023181f,
    0.9163000f, 0.9297995f, 0.9427984f, 0.9552776f, 0.9672179f,
    0.9786000f, 0.9893856f, 0.9995488f, 1.0090892f, 1.0180064f,
    1.0263000f, 1.0339827f, 1.0409860f, 1.0471880f, 1.0524667f,
    1.0567000f, 1.0597944f, 1.0617992f, 1.0628068f, 1.0629096f,
    1.0622000f, 1.0607352f, 1.0584436f, 1.0552244f, 1.0509768f,
    1.0456000f, 1.0390369f, 1.0313608f, 1.0226662f, 1.0130477f,
    1.0026000f, 0.9913675f, 0.9793314f, 0.9664916f, 0.9528479f,
    0.9384000f, 0.9231940f, 0.9072440f, 0.8905020f, 0.8729200f,
    0.8544499f, 0.8350840f, 0.8149460f, 0.7941860f, 0.7729540f,
    0.7514000f, 0.7295836f, 0.7075888f, 0.6856022f, 0.6638104f,
    0.6424000f, 0.6215149f, 0.6011138f, 0.5811052f, 0.5613977f,
    0.5419000f, 0.5225995f, 0.5035464f, 0.4847436f, 0.4661939f,
    0.4479000f, 0.4298613f, 0.4120980f, 0.3946440f, 0.3775333f,
    0.3608000f, 0.3444563f, 0.3285168f, 0.3130192f, 0.2980011f,
    0.2835000f, 0.2695448f, 0.2561184f, 0.2431896f, 0.2307272f,
    0.2187000f, 0.2070971f, 0.1959232f, 0.1851708f, 0.1748323f,
    0.1649000f, 0.1553667f, 0.1462300f, 0.1374900f, 0.1291467f,
    0.1212000f, 0.1136397f, 0.1064650f, 0.09969044f, 0.09333061f,
    0.08740000f, 0.08190096f, 0.07680428f, 0.07207712f, 0.06768664f,
    0.06360000f, 0.05980685f, 0.05628216f, 0.05297104f, 0.04981861f,
    0.04677000f, 0.04378405f, 0.04087536f, 0.03807264f, 0.03540461f,
    0.03290000f, 0.03056419f, 0.02838056f, 0.02634484f, 0.02445275f,
    0.02270000f, 0.02108429f, 0.01959988f, 0.01823732f, 0.01698717f,
    0.01584000f, 0.01479064f, 0.01383132f, 0.01294868f, 0.01212920f,
    0.01135916f, 0.01062935f, 0.009938846f, 0.009288422f, 0.008678854f,
    0.008110916f, 0.007582388f, 0.007088746f, 0.006627313f, 0.006195408f,
    0.005790346f, 0.005409826f, 0.005052583f, 0.004717512f, 0.004403507f,
    0.004109457f, 0.003833913f, 0.003575748f, 0.003334342f, 0.003109075f,
    0.002899327f, 0.002704348f, 0.002523020f, 0.002354168f, 0.002196616f,
    0.002049190f, 0.001910960f, 0.001781438f, 0.001660110f, 0.001546459f,
    0.001439971f, 0.001340042f, 0.001246275f, 0.001158471f, 0.001076430f,
    0.0009999493f, 0.0009287358f, 0.0008624332f, 0.0008007503f, 0.0007433960f,
    0.0006900786f, 0.0006405156f, 0.0005945021f, 0.0005518646f, 0.0005124290f,
    0.0004760213f, 0.0004424536f, 0.0004115117f, 0.0003829814f, 0.0003566491f,
    0.0003323011f, 0.0003097586f, 0.0002888871f, 0.0002695394f, 0.0002515682f,
    0.0002348261f, 0.0002191710f, 0.0002045258f, 0.0001908405f, 0.0001780654f,
    0.0001661505f, 0.0001550236f, 0.0001446219f, 0.0001349098f, 0.0001258520f,
    0.0001174130f, 0.0001095515f, 0.0001022245f, 0.00009539445f, 0.00008902390f,
    0.00008307527f, 0.00007751269f, 0.00007231304f, 0.00006745778f, 0.00006292844f,
    0.00005870652f, 0.00005477028f, 0.00005109918f, 0.00004767654f, 0.00004448567f,
    0.00004150994f, 0.00003873324f, 0.00003614203f, 0.00003372352f, 0.00003146487f,
    0.00002935326f, 0.00002737573f, 0.00002552433f, 0.00002379376f, 0.00002217870f,
    0.00002067383f, 0.00001927226f, 0.00001796640f, 0.00001674991f, 0.00001561648f,
    0.00001455977f, 0.00001357387f, 0.00001265436f, 0.00001179723f, 0.00001099844f,
    0.00001025398f, 0.000009559646f, 0.000008912044f, 0.000008308358f, 0.000007745769f,
    0.000007221456f, 0.000006732475f, 0.000006276423f, 0.000005851304f, 0.000005455118f,
    0.000005085868f, 0.000004741466f, 0.000004420236f, 0.000004120783f, 0.000003841716f,
    0.000003581652f, 0.000003339127f, 0.000003112949f, 0.000002902121f, 0.000002705645f,
    0.000002522525f, 0.000002351726f, 0.000002192415f, 0.000002043902f, 0.000001905497f,
    0.000001776509f, 0.000001656215f, 0.000001544022f, 0.000001439440f, 0.000001341977f,
    0.000001251141f};

static constexpr const std::array cie_y_samples{
    // CIE Y function values
    0.000003917000f, 0.000004393581f, 0.000004929604f, 0.000005532136f, 0.000006208245f,
    0.000006965000f, 0.000007813219f, 0.000008767336f, 0.000009839844f, 0.00001104323f,
    0.00001239000f, 0.00001388641f, 0.00001555728f, 0.00001744296f, 0.00001958375f,
    0.00002202000f, 0.00002483965f, 0.00002804126f, 0.00003153104f, 0.00003521521f,
    0.00003900000f, 0.00004282640f, 0.00004691460f, 0.00005158960f, 0.00005717640f,
    0.00006400000f, 0.00007234421f, 0.00008221224f, 0.00009350816f, 0.0001061361f,
    0.0001200000f, 0.0001349840f, 0.0001514920f, 0.0001702080f, 0.0001918160f,
    0.0002170000f, 0.0002469067f, 0.0002812400f, 0.0003185200f, 0.0003572667f,
    0.0003960000f, 0.0004337147f, 0.0004730240f, 0.0005178760f, 0.0005722187f,
    0.0006400000f, 0.0007245600f, 0.0008255000f, 0.0009411600f, 0.001069880f,
    0.001210000f, 0.001362091f, 0.001530752f, 0.001720368f, 0.001935323f,
    0.002180000f, 0.002454800f, 0.002764000f, 0.003117800f, 0.003526400f,
    0.004000000f, 0.004546240f, 0.005159320f, 0.005829280f, 0.006546160f,
    0.007300000f, 0.008086507f, 0.008908720f, 0.009767680f, 0.01066443f,
    0.01160000f, 0.01257317f, 0.01358272f, 0.01462968f, 0.01571509f,
    0.01684000f, 0.01800736f, 0.01921448f, 0.02045392f, 0.02171824f,
    0.02300000f, 0.02429461f, 0.02561024f, 0.02695857f, 0.02835125f,
    0.02980000f, 0.03131083f, 0.03288368f, 0.03452112f, 0.03622571f,
    0.03800000f, 0.03984667f, 0.04176800f, 0.04376600f, 0.04584267f,
    0.04800000f, 0.05024368f, 0.05257304f, 0.05498056f, 0.05745872f,
    0.06000000f, 0.06260197f, 0.06527752f, 0.06804208f, 0.07091109f,
    0.07390000f, 0.07701600f, 0.08026640f, 0.08366680f, 0.08723280f,
    0.09098000f, 0.09491755f, 0.09904584f, 0.1033674f, 0.1078846f,
    0.1126000f, 0.1175320f, 0.1226744f, 0.1279928f, 0.1334528f,
    0.1390200f, 0.1446764f, 0.1504693f, 0.1564619f, 0.1627177f,
    0.1693000f, 0.1762431f, 0.1835581f, 0.1912735f, 0.1994180f,
    0.2080200f, 0.2171199f, 0.2267345f, 0.2368571f, 0.2474812f,
    0.2586000f, 0.2701849f, 0.2822939f, 0.2950505f, 0.3085780f,
    0.3230000f, 0.3384021f, 0.3546858f, 0.3716986f, 0.3892875f,
    0.4073000f, 0.4256299f, 0.4443096f, 0.4633944f, 0.4829395f,
    0.5030000f, 0.5235693f, 0.5445120f, 0.5656900f, 0.5869653f,
    0.6082000f, 0.6293456f, 0.6503068f, 0.6708752f, 0.6908424f,
    0.7100000f, 0.7281852f, 0.7454636f, 0.7619694f, 0.7778368f,
    0.7932000f, 0.8081104f, 0.8224962f, 0.8363068f, 0.8494916f,
    0.8620000f, 0.8738108f, 0.8849624f, 0.8954936f, 0.9054432f,
    0.9148501f, 0.9237348f, 0.9320924f, 0.9399226f, 0.9472252f,
    0.9540000f, 0.9602561f, 0.9660074f, 0.9712606f, 0.9760225f,
    0.9803000f, 0.9840924f, 0.9874812f, 0.9903128f, 0.9928116f,
    0.9949501f, 0.9967108f, 0.9980983f, 0.9991120f, 0.9997482f,
    1.0000000f, 0.9998567f, 0.9993046f, 0.9983255f, 0.9968987f,
    0.9950000f, 0.9926005f, 0.9897426f, 0.9864444f, 0.9827241f,
    0.9786000f, 0.9740837f, 0.9691712f, 0.9638568f, 0.9581349f,
    0.9520000f, 0.9454504f, 0.9384992f, 0.9311628f, 0.9234576f,
    0.9154000f, 0.9070064f, 0.8982772f, 0.8892048f, 0.8797816f,
    0.8700000f, 0.8598613f, 0.8493920f, 0.8386220f, 0.8275813f,
    0.8163000f, 0.8047947f, 0.7930820f, 0.7811920f, 0.7691547f,
    0.7570000f, 0.7447541f, 0.7324224f, 0.7200036f, 0.7074965f,
    0.6949000f, 0.6822192f, 0.6694716f, 0.6566744f, 0.6438448f,
    0.6310000f, 0.6181555f, 0.6053144f, 0.5924756f, 0.5796379f,
    0.5668000f, 0.5539611f, 0.5411372f, 0.5283528f, 0.5156323f,
    0.5030000f, 0.4904688f, 0.4780304f, 0.4656776f, 0.4534032f,
    0.4412000f, 0.4290800f, 0.4170360f, 0.4050320f, 0.3930320f,
    0.3810000f, 0.3689184f, 0.3568272f, 0.3447768f, 0.3328176f,
    0.3210000f, 0.3093381f, 0.2978504f, 0.2865936f, 0.2756245f,
    0.2650000f, 0.2547632f, 0.2448896f, 0.2353344f, 0.2260528f,
    0.2170000f, 0.2081616f, 0.1995488f, 0.1911552f, 0.1829744f,
    0.1750000f, 0.1672235f, 0.1596464f, 0.1522776f, 0.1451259f,
    0.1382000f, 0.1315003f, 0.1250248f, 0.1187792f, 0.1127691f,
    0.1070000f, 0.1014762f, 0.09618864f, 0.09112296f, 0.08626485f,
    0.08160000f, 0.07712064f, 0.07282552f, 0.06871008f, 0.06476976f,
    0.06100000f, 0.05739621f, 0.05395504f, 0.05067376f, 0.04754965f,
    0.04458000f, 0.04175872f, 0.03908496f, 0.03656384f, 0.03420048f,
    0.03200000f, 0.02996261f, 0.02807664f, 0.02632936f, 0.02470805f,
    0.02320000f, 0.02180077f, 0.02050112f, 0.01928108f, 0.01812069f,
    0.01700000f, 0.01590379f, 0.01483718f, 0.01381068f, 0.01283478f,
    0.01192000f, 0.01106831f, 0.01027339f, 0.009533311f, 0.008846157f,
    0.008210000f, 0.007623781f, 0.007085424f, 0.006591476f, 0.006138485f,
    0.005723000f, 0.005343059f, 0.004995796f, 0.004676404f, 0.004380075f,
    0.004102000f, 0.003838453f, 0.003589099f, 0.003354219f, 0.003134093f,
    0.002929000f, 0.002738139f, 0.002559876f, 0.002393244f, 0.002237275f,
    0.002091000f, 0.001953587f, 0.001824580f, 0.001703580f, 0.001590187f,
    0.001484000f, 0.001384496f, 0.001291268f, 0.001204092f, 0.001122744f,
    0.001047000f, 0.0009765896f, 0.0009111088f, 0.0008501332f, 0.0007932384f,
    0.0007400000f, 0.0006900827f, 0.0006433100f, 0.0005994960f, 0.0005584547f,
    0.0005200000f, 0.0004839136f, 0.0004500528f, 0.0004183452f, 0.0003887184f,
    0.0003611000f, 0.0003353835f, 0.0003114404f, 0.0002891656f, 0.0002684539f,
    0.0002492000f, 0.0002313019f, 0.0002146856f, 0.0001992884f, 0.0001850475f,
    0.0001719000f, 0.0001597781f, 0.0001486044f, 0.0001383016f, 0.0001287925f,
    0.0001200000f, 0.0001118595f, 0.0001043224f, 0.00009733560f, 0.00009084587f,
    0.00008480000f, 0.00007914667f, 0.00007385800f, 0.00006891600f, 0.00006430267f,
    0.00006000000f, 0.00005598187f, 0.00005222560f, 0.00004871840f, 0.00004544747f,
    0.00004240000f, 0.00003956104f, 0.00003691512f, 0.00003444868f, 0.00003214816f,
    0.00003000000f, 0.00002799125f, 0.00002611356f, 0.00002436024f, 0.00002272461f,
    0.00002120000f, 0.00001977855f, 0.00001845285f, 0.00001721687f, 0.00001606459f,
    0.00001499000f, 0.00001398728f, 0.00001305155f, 0.00001217818f, 0.00001136254f,
    0.00001060000f, 0.000009885877f, 0.000009217304f, 0.000008592362f, 0.000008009133f,
    0.000007465700f, 0.000006959567f, 0.000006487995f, 0.000006048699f, 0.000005639396f,
    0.000005257800f, 0.000004901771f, 0.000004569720f, 0.000004260194f, 0.000003971739f,
    0.000003702900f, 0.000003452163f, 0.000003218302f, 0.000003000300f, 0.000002797139f,
    0.000002607800f, 0.000002431220f, 0.000002266531f, 0.000002113013f, 0.000001969943f,
    0.000001836600f, 0.000001712230f, 0.000001596228f, 0.000001488090f, 0.000001387314f,
    0.000001293400f, 0.000001205820f, 0.000001124143f, 0.000001048009f, 0.0000009770578f,
    0.0000009109300f, 0.0000008492513f, 0.0000007917212f, 0.0000007380904f, 0.0000006881098f,
    0.0000006415300f, 0.0000005980895f, 0.0000005575746f, 0.0000005198080f, 0.0000004846123f,
    0.0000004518100f};

static constexpr const std::array cie_z_samples{
    // CIE Z function values
    0.0006061000f, 0.0006808792f, 0.0007651456f, 0.0008600124f, 0.0009665928f,
    0.001086000f, 0.001220586f, 0.001372729f, 0.001543579f, 0.001734286f,
    0.001946000f, 0.002177777f, 0.002435809f, 0.002731953f, 0.003078064f,
    0.003486000f, 0.003975227f, 0.004540880f, 0.005158320f, 0.005802907f,
    0.006450001f, 0.007083216f, 0.007745488f, 0.008501152f, 0.009414544f,
    0.01054999f, 0.01196580f, 0.01365587f, 0.01558805f, 0.01773015f,
    0.02005001f, 0.02251136f, 0.02520288f, 0.02827972f, 0.03189704f,
    0.03621000f, 0.04143771f, 0.04750372f, 0.05411988f, 0.06099803f,
    0.06785001f, 0.07448632f, 0.08136156f, 0.08915364f, 0.09854048f,
    0.1102000f, 0.1246133f, 0.1417017f, 0.1613035f, 0.1832568f,
    0.2074000f, 0.2336921f, 0.2626114f, 0.2947746f, 0.3307985f,
    0.3713000f, 0.4162091f, 0.4654642f, 0.5196948f, 0.5795303f,
    0.6456000f, 0.7184838f, 0.7967133f, 0.8778459f, 0.9594390f,
    1.0390501f, 1.1153673f, 1.1884971f, 1.2581233f, 1.3239296f,
    1.3856000f, 1.4426352f, 1.4948035f, 1.5421903f, 1.5848807f,
    1.6229600f, 1.6564048f, 1.6852959f, 1.7098745f, 1.7303821f,
    1.7470600f, 1.7600446f, 1.7696233f, 1.7762637f, 1.7804334f,
    1.7826000f, 1.7829682f, 1.7816998f, 1.7791982f, 1.7758671f,
    1.7721100f, 1.7682589f, 1.7640390f, 1.7589438f, 1.7524663f,
    1.7441000f, 1.7335595f, 1.7208581f, 1.7059369f, 1.6887372f,
    1.6692000f, 1.6475287f, 1.6234127f, 1.5960223f, 1.5645280f,
    1.5281000f, 1.4861114f, 1.4395215f, 1.3898799f, 1.3387362f,
    1.2876400f, 1.2374223f, 1.1878243f, 1.1387611f, 1.0901480f,
    1.0419000f, 0.9941976f, 0.9473473f, 0.9014531f, 0.8566193f,
    0.8129501f, 0.7705173f, 0.7294448f, 0.6899136f, 0.6521049f,
    0.6162000f, 0.5823286f, 0.5504162f, 0.5203376f, 0.4919673f,
    0.4651800f, 0.4399246f, 0.4161836f, 0.3938822f, 0.3729459f,
    0.3533000f, 0.3348578f, 0.3175521f, 0.3013375f, 0.2861686f,
    0.2720000f, 0.2588171f, 0.2464838f, 0.2347718f, 0.2234533f,
    0.2123000f, 0.2011692f, 0.1901196f, 0.1792254f, 0.1685608f,
    0.1582000f, 0.1481383f, 0.1383758f, 0.1289942f, 0.1200751f,
    0.1117000f, 0.1039048f, 0.09666748f, 0.08998272f, 0.08384531f,
    0.07824999f, 0.07320899f, 0.06867816f, 0.06456784f, 0.06078835f,
    0.05725001f, 0.05390435f, 0.05074664f, 0.04775276f, 0.04489859f,
    0.04216000f, 0.03950728f, 0.03693564f, 0.03445836f, 0.03208872f,
    0.02984000f, 0.02771181f, 0.02569444f, 0.02378716f, 0.02198925f,
    0.02030000f, 0.01871805f, 0.01724036f, 0.01586364f, 0.01458461f,
    0.01340000f, 0.01230723f, 0.01130188f, 0.01037792f, 0.009529306f,
    0.008749999f, 0.008035200f, 0.007381600f, 0.006785400f, 0.006242800f,
    0.005749999f, 0.005303600f, 0.004899800f, 0.004534200f, 0.004202400f,
    0.003900000f, 0.003623200f, 0.003370600f, 0.003141400f, 0.002934800f,
    0.002749999f, 0.002585200f, 0.002438600f, 0.002309400f, 0.002196800f,
    0.002100000f, 0.002017733f, 0.001948200f, 0.001889800f, 0.001840933f,
    0.001800000f, 0.001766267f, 0.001737800f, 0.001711200f, 0.001683067f,
    0.001650001f, 0.001610133f, 0.001564400f, 0.001513600f, 0.001458533f,
    0.001400000f, 0.001336667f, 0.001270000f, 0.001205000f, 0.001146667f,
    0.001100000f, 0.001068800f, 0.001049400f, 0.001035600f, 0.001021200f,
    0.001000000f, 0.0009686400f, 0.0009299200f, 0.0008868800f, 0.0008425600f,
    0.0008000000f, 0.0007609600f, 0.0007236800f, 0.0006859200f, 0.0006454400f,
    0.0006000000f, 0.0005478667f, 0.0004916000f, 0.0004354000f, 0.0003834667f,
    0.0003400000f, 0.0003072533f, 0.0002831600f, 0.0002654400f, 0.0002518133f,
    0.0002400000f, 0.0002295467f, 0.0002206400f, 0.0002119600f, 0.0002021867f,
    0.0001900000f, 0.0001742133f, 0.0001556400f, 0.0001359600f, 0.0001168533f,
    0.0001000000f, 0.00008613333f, 0.00007460000f, 0.00006500000f, 0.00005693333f,
    0.00004999999f, 0.00004416000f, 0.00003948000f, 0.00003572000f, 0.00003264000f,
    0.00003000000f, 0.00002765333f, 0.00002556000f, 0.00002364000f, 0.00002181333f,
    0.00002000000f, 0.00001813333f, 0.00001620000f, 0.00001420000f, 0.00001213333f,
    0.00001000000f, 0.000007733333f, 0.000005400000f, 0.000003200000f, 0.000001333333f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f, 0.000000000000f,
    0.000000000000f};

static_assert(cie_x_samples.size() == DenselySampledSpectrum::sample_count);
static_assert(cie_y_samples.size() == DenselySampledSpectrum::sample_count);
static_assert(cie_z_samples.size() == DenselySampledSpectrum::sample_count);

constexpr const std::array cie_illum_d6500_samples{
    0.47161809932375853f, 0.47713008171230636f, 0.48264206410085414f, 0.4881540464894019f, 0.4936660288779497f,
    0.4991780112664976f, 0.5046899936550453f, 0.5102019760435932f, 0.515713958432141f, 0.5212259408206888f,
    0.5267379232092366f, 0.5246005963443631f, 0.5224632694794894f, 0.5203259426146157f, 0.5181886157497422f,
    0.5160512888848685f, 0.513913962019995f, 0.5117766351551213f, 0.5096393082902477f, 0.5075019814253741f,
    0.5053646545605004f, 0.5100897037934252f, 0.5148147530263502f, 0.519539802259275f, 0.5242648514921999f,
    0.5289899007251249f, 0.5337151582703688f, 0.5384404158156127f, 0.5431656733608566f, 0.5478909309061006f,
    0.5526161884513445f, 0.5810382787907761f, 0.6094603691302076f, 0.6378824594696391f, 0.6663045498090706f,
    0.6947266401485027f, 0.7231489307104546f, 0.7515712212724063f, 0.779993511834358f, 0.8084158023963098f,
    0.8368380929582621f, 0.8456670991438475f, 0.8544961053294329f, 0.8633251115150179f, 0.8721541177006032f,
    0.8809831238861885f, 0.8898123323167438f, 0.8986415407472991f, 0.9074707491778542f, 0.9162999576084094f,
    0.9251291660389646f, 0.927096807351601f, 0.9290644486642374f, 0.9310320899768739f, 0.9329997312895104f,
    0.9349673726021469f, 0.936935015937233f, 0.9389026592723194f, 0.9408703026074055f, 0.9428379459424916f,
    0.9448055892775777f, 0.9379802219868303f, 0.9311548546960827f, 0.9243294874053354f, 0.9175041201145878f,
    0.9106787528238404f, 0.9038535877780629f, 0.8970284227322853f, 0.8902032576865077f, 0.8833780926407302f,
    0.8765529275949526f, 0.8949396265704007f, 0.9133263255458487f, 0.931713024521297f, 0.9500997234967449f,
    0.968486422472193f, 0.9868733216701611f, 1.0052602208681294f, 1.023647120066098f, 1.0420340192640662f,
    1.0604209184620343f, 1.0726992085652052f, 1.0849774986683758f, 1.0972557887715464f, 1.1095340788747172f,
    1.121812368977888f, 1.134092695687906f, 1.1463730223979243f, 1.1586533491079423f, 1.1709336758179605f,
    1.1832140025279787f, 1.1840270293296242f, 1.1848400561312702f, 1.1856530829329162f, 1.1864661097345617f,
    1.1872791365362076f, 1.188092145135806f, 1.1889051537354045f, 1.189718162335003f, 1.1905311709346016f,
    1.1913441795342f, 1.1883590498450978f, 1.1853739201559954f, 1.1823887904668935f, 1.1794036607777914f,
    1.176418531088689f, 1.173435419804387f, 1.170452308520085f, 1.1674691972357827f, 1.1644860859514807f,
    1.1615029746671786f, 1.1625768914127097f, 1.163650808158241f, 1.1647247249037724f, 1.1657986416493036f,
    1.1668725583948347f, 1.1679464771628156f, 1.1690203959307968f, 1.1700943146987777f, 1.1711682334667586f,
    1.1722421522347395f, 1.1650503211033436f, 1.157858489971947f, 1.1506666588405512f, 1.1434748277091549f,
    1.1362829965777586f, 1.1290911654463625f, 1.1218993343149664f, 1.1147075031835703f, 1.107515672052174f,
    1.1003238409207776f, 1.100871932879139f, 1.1014200248375003f, 1.1019681167958617f, 1.102516208754223f,
    1.1030643007125844f, 1.1036143969185979f, 1.1041644931246113f, 1.1047145893306247f, 1.1052646855366384f,
    1.105814781742652f, 1.1042453749329935f, 1.102675968123335f, 1.1011065613136766f, 1.099537154504018f,
    1.0979677476943597f, 1.0963983247051035f, 1.0948289017158475f, 1.0932594787265915f, 1.0916900557373352f,
    1.0901206327480792f, 1.0870748154108445f, 1.0840289980736095f, 1.0809831807363748f, 1.0779373633991396f,
    1.0748915460619048f, 1.071845742881818f, 1.068799939701731f, 1.065754136521644f, 1.062708333341557f,
    1.0596625301614702f, 1.062593053708686f, 1.065523577255902f, 1.0684541008031176f, 1.0713846243503335f,
    1.0743151478975494f, 1.0772477100740625f, 1.0801802722505753f, 1.0831128344270884f, 1.0860453966036014f,
    1.0889779587801143f, 1.0856570842388489f, 1.0823362096975837f, 1.0790153351563185f, 1.0756944606150531f,
    1.072373586073788f, 1.0690527277121202f, 1.0657318693504527f, 1.062411010988785f, 1.059090152627117f,
    1.0557692942654495f, 1.0554052512970913f, 1.0550412083287328f, 1.0546771653603744f, 1.0543131223920161f,
    1.0539490794236577f, 1.0535870609274485f, 1.0532250424312393f, 1.05286302393503f, 1.0525010054388206f,
    1.0521389869426117f, 1.0480475813127863f, 1.043956175682961f, 1.0398647700531356f, 1.0357733644233105f,
    1.0316819587934851f, 1.027590536984062f, 1.023499115174639f, 1.019407693365216f, 1.0153162715557935f,
    1.0112248497463705f, 1.0075178996697205f, 1.0038109495930705f, 1.0001039995164205f, 0.9963970494397706f,
    0.9926900993631207f, 0.9889831492864707f, 0.9852761992098209f, 0.9815692491331708f, 0.9778622990565211f,
    0.974155348979871f, 0.9736030139220403f, 0.9730506788642094f, 0.9724983438063785f, 0.9719460087485475f,
    0.9713936736907167f, 0.9708413548124835f, 0.9702890359342502f, 0.9697367170560169f, 0.9691843981777837f,
    0.9686320792995504f, 0.9614499539042619f, 0.9542678285089737f, 0.9470857031136852f, 0.9399035777183972f,
    0.9327214523231088f, 0.9255393269278206f, 0.9183572015325322f, 0.9111750761372437f, 0.9039929507419555f,
    0.8968108253466671f, 0.8981462509056918f, 0.8994816764647166f, 0.9008171020237414f, 0.902152527582766f,
    0.9034879531417909f, 0.9048233807232653f, 0.9061588083047398f, 0.9074942358862143f, 0.9088296634676887f,
    0.9101650910491632f, 0.9097533081556481f, 0.9093415252621335f, 0.9089297423686185f, 0.9085179594751036f,
    0.9081061765815887f, 0.9076946100901916f, 0.9072830435987944f, 0.9068714771073974f, 0.9064599106160003f,
    0.9060483441246032f, 0.9041266225323938f, 0.9022049009401842f, 0.9002831793479747f, 0.8983614577557653f,
    0.8964397361635558f, 0.8945179983917486f, 0.8925962606199416f, 0.8906745228481345f, 0.8887527850763274f,
    0.8868310473045202f, 0.882371341449719f, 0.8779116355949181f, 0.8734519297401169f, 0.8689922238853157f,
    0.8645325180305146f, 0.8600730123982335f, 0.8556135067659528f, 0.8511540011336719f, 0.846694495501391f,
    0.8422349898691099f, 0.8426501967699662f, 0.8430654036708224f, 0.8434806105716786f, 0.8438958174725347f,
    0.8443110243733909f, 0.8447262474538447f, 0.8451414705342983f, 0.8455566936147522f, 0.8459719166952058f,
    0.8463871397756596f, 0.8426735095476522f, 0.838959879319645f, 0.8352462490916376f, 0.8315326188636302f,
    0.8278189886356228f, 0.8241053725647632f, 0.8203917564939041f, 0.8166781404230444f, 0.8129645243521849f,
    0.8092509082813254f, 0.8094408061958592f, 0.8096307041103931f, 0.809820602024927f, 0.810010499939461f,
    0.8102003978539948f, 0.8103903119481264f, 0.8105802260422578f, 0.8107701401363892f, 0.8109600542305206f,
    0.8111499683246521f, 0.8132363294570985f, 0.815322690589545f, 0.8174090517219913f, 0.8194954128544376f,
    0.8215817739868841f, 0.8236681351193303f, 0.8257544962517768f, 0.8278408573842233f, 0.8299272185166696f,
    0.8320135796491162f, 0.8279751439993706f, 0.8239367083496245f, 0.8198982726998786f, 0.8158598370501327f,
    0.8118214014003866f, 0.8077829839526883f, 0.8037445665049896f, 0.799706149057291f, 0.7956677316095924f,
    0.7916293141618936f, 0.7829701917286166f, 0.7743110692953388f, 0.7656519468620612f, 0.7569928244287835f,
    0.7483337019955059f, 0.739674779784749f, 0.7310158575739917f, 0.7223569353632341f, 0.7136980131524766f,
    0.7050390909417192f, 0.70694808323552f, 0.708857075529321f, 0.7107660678231218f, 0.7126750601169228f,
    0.7145840524107236f, 0.7164930447045246f, 0.7184020369983254f, 0.7203110292921264f, 0.7222200215859275f,
    0.7241290138797282f, 0.7268995596332645f, 0.7296701053868009f, 0.7324406511403372f, 0.7352111968938735f,
    0.7379817426474097f, 0.7407525048030639f, 0.7435232669587182f, 0.7462940291143724f, 0.7490647912700265f,
    0.7518355534256805f, 0.7389474967605622f, 0.726059440095444f, 0.7131713834303257f, 0.7002833267652073f,
    0.68739527010009f, 0.6745072073676226f, 0.6616191446351553f, 0.6487310819026878f, 0.6358430191702205f,
    0.622954956437754f, 0.6313295100860646f, 0.6397040637343752f, 0.6480786173826858f, 0.6564531710309963f,
    0.6648277246793063f, 0.673202284394966f, 0.6815768441106257f, 0.6899514038262853f, 0.698325963541945f,
    0.706700523257604f, 0.7119603102135247f, 0.7172200971694451f, 0.7224798841253657f, 0.7277396710812862f,
    0.7329994580372063f, 0.7382592429706772f, 0.743519027904148f, 0.7487788128376189f, 0.7540385977710894f,
    0.7592983827045602f, 0.7476749597916353f, 0.7360515368787105f, 0.7244281139657857f, 0.7128046910528616f,
    0.7011812681399369f, 0.6895580535393311f, 0.6779348389387253f, 0.6663116243381194f, 0.6546884097375144f,
    0.6430651951369086f, 0.625697814854904f, 0.6083304345728995f, 0.5909630542908949f, 0.5735956740088916f,
    0.556228293726887f, 0.5388611116449531f, 0.521493929563019f, 0.504126747481085f, 0.4867595653991522f,
    0.4693923833172183f, 0.49000842657396776f, 0.5106244698307173f, 0.5312405130874668f, 0.5518565563442148f,
    0.5724725996009643f, 0.593088634767915f, 0.6137046699348656f, 0.6343207051018164f, 0.6549367402687656f,
    0.6755527754357162f, 0.672091765354773f, 0.6686307552738299f, 0.6651697451928867f, 0.6617087351119437f,
    0.6582477250310006f, 0.6547867068602585f, 0.6513256886895167f, 0.6478646705187746f, 0.644403652348033f,
    0.640942634177291f, 0.6418741704639779f, 0.6428057067506648f, 0.6437372430373518f, 0.6446687793240389f,
    0.6456003156107257f, 0.6465318599872116f, 0.6474634043636973f, 0.6483949487401831f, 0.6493264931166688f,
    0.6502580374931546f, 0.6453513682323158f, 0.640444698971477f, 0.6355380297106381f, 0.6306313604497997f,
    0.6257246911889609f, 0.6208182302404411f, 0.6159117692919214f, 0.6110053083434019f, 0.6060988473948822f,
    0.6011923864463624f, 0.593615276624763f, 0.5860381668031637f, 0.5784610569815648f, 0.5708839471599656f,
    0.5633068373383662f, 0.5557299338066362f, 0.5481530302749061f, 0.5405761267431767f, 0.5329992232114467f,
    0.5254223196797166f, 0.5309654477936367f, 0.5365085759075569f, 0.5420517040214765f, 0.5475948321353966f,
    0.5531379602493166f, 0.5586810944305859f, 0.564224228611855f, 0.5697673627931238f, 0.575310496974393f,
    0.5808536311556621f, 0.5837576626342645f, 0.5866616941128667f, 0.5895657255914688f, 0.592469757070071f,
    0.5953737885486733f, 0.5982780283395946f, 0.6011822681305159f, 0.6040865079214371f, 0.6069907477123584f,
    0.6098949875032796f};

static_assert(cie_illum_d6500_samples.size() == DenselySampledSpectrum::sample_count);

const DenselySampledSpectrum &DenselySampledSpectrum::cie_x() noexcept {
    static DenselySampledSpectrum s{cie_x_samples};
    return s;
}

const DenselySampledSpectrum &DenselySampledSpectrum::cie_y() noexcept {
    static DenselySampledSpectrum s{cie_y_samples};
    return s;
}

const DenselySampledSpectrum &DenselySampledSpectrum::cie_z() noexcept {
    static DenselySampledSpectrum s{cie_z_samples};
    return s;
}

const DenselySampledSpectrum &DenselySampledSpectrum::cie_illum_d65() noexcept {
    static DenselySampledSpectrum s{cie_illum_d6500_samples};
    return s;
}

luisa::compute::Float DenselySampledSpectrum::sample(Expr<float> lambda) const noexcept {
    using namespace luisa::compute;
    auto t = clamp(lambda, visible_wavelength_min, visible_wavelength_max) -
             visible_wavelength_min;
    auto i = cast<uint>(min(t, static_cast<float>(sample_count - 2u)));
    auto s0 = _values[i];
    auto s1 = _values[i + 1u];
    return lerp(s0, s1, fract(t));
}

float DenselySampledSpectrum::cie_y_integral() noexcept {
    static constexpr auto integral = [] {
        auto sum = 0.0;
        for (auto i = 0u; i < sample_count - 1u; i++) {
            sum += 0.5f * (cie_y_samples[i] + cie_y_samples[i + 1u]);
        }
        return static_cast<float>(sum);
    }();
    return integral;
}

SampledSpectrum zero_if_any_nan(const SampledSpectrum &t) noexcept {
    auto has_nan = t.any([](const auto &value) { return isnan(value); });
    return t.map([&has_nan](auto, auto x) noexcept { return ite(has_nan, 0.f, x); });
}

SampledSpectrum ite(const SampledSpectrum &p, const SampledSpectrum &t, const SampledSpectrum &f) noexcept {
    return p.map([&t, &f](auto i, auto b) noexcept { return ite(b != 0.f, t[i], f[i]); });
}

SampledSpectrum ite(Expr<bool> p, const SampledSpectrum &t, const SampledSpectrum &f) noexcept {
    return t.map([p, &f](auto i, auto x) noexcept { return ite(p, x, f[i]); });
}

}// namespace luisa::render
