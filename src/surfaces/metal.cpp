//
// Created by Mike Smith on 2022/1/9.
//

#include <util/sampling.h>
#include <util/scattering.h>
#include <base/surface.h>
#include <base/interaction.h>
#include <base/pipeline.h>
#include <base/scene.h>

#include <utility>

namespace luisa::render {

namespace ior {

static constexpr auto lut_step = 10u;
static constexpr auto lut_min = static_cast<uint>(visible_wavelength_min);
static constexpr auto lut_max = static_cast<uint>(visible_wavelength_max);
static constexpr auto lut_size = (lut_max - lut_min) / lut_step + 1;

std::array Ag{make_float2(0.1937697969f, 1.542775784f), make_float2(0.1919493526f, 1.641277522f), make_float2(0.1985622426f, 1.718246893f), make_float2(0.1878545676f, 1.838499847f),
              make_float2(0.1729970816f, 1.950617723f), make_float2(0.1728184204f, 2.070960395f), make_float2(0.1669451194f, 2.183143955f), make_float2(0.1595063655f, 2.282830594f),
              make_float2(0.157540771f, 2.374538697f), make_float2(0.1516909674f, 2.470420468f), make_float2(0.1433830656f, 2.567380701f), make_float2(0.136052175f, 2.658984755f),
              make_float2(0.1314127294f, 2.746217439f), make_float2(0.1301526264f, 2.830014841f), make_float2(0.1299751091f, 2.91763738f), make_float2(0.1299611827f, 3.009739205f),
              make_float2(0.1300194835f, 3.097640223f), make_float2(0.1298398444f, 3.178399908f), make_float2(0.1286377125f, 3.25768637f), make_float2(0.1247768295f, 3.339599698f),
              make_float2(0.1212451994f, 3.421004641f), make_float2(0.1196626603f, 3.500944672f), make_float2(0.1197220763f, 3.57951429f), make_float2(0.1209507524f, 3.656898309f),
              make_float2(0.1239997394f, 3.73082718f), make_float2(0.1273912211f, 3.805363891f), make_float2(0.1310255041f, 3.880650044f), make_float2(0.1342497102f, 3.962827979f),
              make_float2(0.1370939161f, 4.045474725f), make_float2(0.1394098384f, 4.128773219f), make_float2(0.1400266186f, 4.210472834f), make_float2(0.14006234f, 4.291625677f),
              make_float2(0.1400297838f, 4.370563658f), make_float2(0.1402627758f, 4.448910495f), make_float2(0.1424543348f, 4.523224315f), make_float2(0.1445817677f, 4.597457767f),
              make_float2(0.1463511613f, 4.671242489f), make_float2(0.1479344888f, 4.745183096f), make_float2(0.1469729757f, 4.821255743f), make_float2(0.1460114627f, 4.89732839f),
              make_float2(0.144824417f, 4.974585793f), make_float2(0.1436000683f, 5.052039155f), make_float2(0.1429108603f, 5.130541219f), make_float2(0.1427360381f, 5.210051314f),
              make_float2(0.1425723274f, 5.28955854f), make_float2(0.1434863209f, 5.368787566f), make_float2(0.1444003145f, 5.448016592f), make_float2(0.1458370844f, 5.526618752f)};

std::array Al{make_float2(0.3970816731f, 4.372694111f), make_float2(0.418897724f, 4.492555044f), make_float2(0.4421600534f, 4.614741667f), make_float2(0.4660888111f, 4.740831653f),
              make_float2(0.4901259804f, 4.860607995f), make_float2(0.5148594252f, 4.98035513f), make_float2(0.5393679384f, 5.105134767f), make_float2(0.5643557083f, 5.229618068f),
              make_float2(0.5905420465f, 5.347553738f), make_float2(0.6179529129f, 5.469422816f), make_float2(0.6462718721f, 5.590148464f), make_float2(0.6748840364f, 5.716104038f),
              make_float2(0.7046480176f, 5.83859207f), make_float2(0.7359216949f, 5.959791837f), make_float2(0.7686498231f, 6.078175846f), make_float2(0.8028348113f, 6.198608797f),
              make_float2(0.8386797108f, 6.323296574f), make_float2(0.8764468509f, 6.447290903f), make_float2(0.9159949541f, 6.561749865f), make_float2(0.9585690152f, 6.686967692f),
              make_float2(1.00377683f, 6.807307448f), make_float2(1.049461979f, 6.923654947f), make_float2(1.09697412f, 7.036373115f), make_float2(1.147956533f, 7.145621142f),
              make_float2(1.196662613f, 7.2566574f), make_float2(1.247508671f, 7.368455925f), make_float2(1.300447877f, 7.48084599f), make_float2(1.357068081f, 7.587795265f),
              make_float2(1.415004287f, 7.692507287f), make_float2(1.474770023f, 7.794109039f), make_float2(1.536476938f, 7.900278348f), make_float2(1.598847682f, 8.008009632f),
              make_float2(1.67381387f, 8.115448386f), make_float2(1.750357498f, 8.220966763f), make_float2(1.836739375f, 8.312429926f), make_float2(1.92676609f, 8.403121288f),
              make_float2(2.037143236f, 8.489503402f), make_float2(2.148323782f, 8.571541441f), make_float2(2.270492494f, 8.594165277f), make_float2(2.392661206f, 8.616789112f),
              make_float2(2.490392437f, 8.612691597f), make_float2(2.584081727f, 8.604174388f), make_float2(2.664676766f, 8.569402853f), make_float2(2.732685401f, 8.509395235f),
              make_float2(2.799770355f, 8.449464163f), make_float2(2.777267437f, 8.396957353f), make_float2(2.754764518f, 8.344450542f), make_float2(2.720695662f, 8.297934788f)};

std::array Au{make_float2(1.726248938f, 1.85351432f), make_float2(1.706064787f, 1.882606367f), make_float2(1.687649576f, 1.918247288f), make_float2(1.670683654f, 1.940870883f),
              make_float2(1.65791634f, 1.956026265f), make_float2(1.641447387f, 1.958665792f), make_float2(1.62832588f, 1.951644869f), make_float2(1.60950048f, 1.934899111f),
              make_float2(1.574347605f, 1.911390537f), make_float2(1.508458089f, 1.878849833f), make_float2(1.418607767f, 1.843105381f), make_float2(1.321224482f, 1.810205112f),
              make_float2(1.189900705f, 1.796461427f), make_float2(1.020243859f, 1.813977192f), make_float2(0.8511633401f, 1.886770717f), make_float2(0.6997216254f, 2.01763491f),
              make_float2(0.5720483809f, 2.183785131f), make_float2(0.4729087806f, 2.371122542f), make_float2(0.3975742176f, 2.55493212f), make_float2(0.3504088361f, 2.714063108f),
              make_float2(0.3172423721f, 2.837485333f), make_float2(0.2871566093f, 2.909768875f), make_float2(0.259956335f, 2.947748068f), make_float2(0.2369232091f, 2.969142474f),
              make_float2(0.2202513683f, 2.999997394f), make_float2(0.2060566022f, 3.030473627f), make_float2(0.1939191333f, 3.060049764f), make_float2(0.1836960409f, 3.066340898f),
              make_float2(0.1748119963f, 3.090003429f), make_float2(0.1677895223f, 3.137816018f), make_float2(0.1638874119f, 3.274985782f), make_float2(0.1610523781f, 3.442713186f),
              make_float2(0.1604727581f, 3.632554764f), make_float2(0.1600533107f, 3.81752775f), make_float2(0.1604979233f, 3.963709768f), make_float2(0.1611028001f, 4.107318852f),
              make_float2(0.1626024896f, 4.236562331f), make_float2(0.1641695585f, 4.364805488f), make_float2(0.1666581804f, 4.479367161f), make_float2(0.1691468023f, 4.593928835f),
              make_float2(0.1718752311f, 4.701474343f), make_float2(0.1746433237f, 4.807859383f), make_float2(0.1770963978f, 4.914220796f), make_float2(0.1792466708f, 5.020559496f),
              make_float2(0.1814003566f, 5.126824152f), make_float2(0.1838850538f, 5.225907128f), make_float2(0.1863697511f, 5.324990104f), make_float2(0.1892216027f, 5.419107323f)};

std::array Cu{make_float2(1.280194277f, 1.933855609f), make_float2(1.249454471f, 1.972017413f), make_float2(1.206443503f, 2.094096699f), make_float2(1.177668194f, 2.196398007f),
              make_float2(1.175019456f, 2.13023396f), make_float2(1.17878947f, 2.185819898f), make_float2(1.178201378f, 2.248311125f), make_float2(1.174279952f, 2.301176317f),
              make_float2(1.171013765f, 2.349021495f), make_float2(1.165807858f, 2.393856878f), make_float2(1.159567491f, 2.436338568f), make_float2(1.154051412f, 2.477448453f),
              make_float2(1.147587828f, 2.514696715f), make_float2(1.139769271f, 2.546829525f), make_float2(1.133619762f, 2.574287551f), make_float2(1.127238808f, 2.595373925f),
              make_float2(1.111475832f, 2.602081192f), make_float2(1.081428899f, 2.592924859f), make_float2(1.032979456f, 2.582490839f), make_float2(0.9581618042f, 2.577064733f),
              make_float2(0.8613992414f, 2.592596117f), make_float2(0.7375526842f, 2.63782205f), make_float2(0.602742548f, 2.709812819f), make_float2(0.4731925953f, 2.805177608f),
              make_float2(0.3901734284f, 2.943488315f), make_float2(0.3239992232f, 3.089579534f), make_float2(0.2717950365f, 3.241085789f), make_float2(0.2458837437f, 3.378350149f),
              make_float2(0.2266090293f, 3.51114933f), make_float2(0.216560616f, 3.637740929f), make_float2(0.2119930413f, 3.751598715f), make_float2(0.2092997592f, 3.861101896f),
              make_float2(0.2112271572f, 3.961653335f), make_float2(0.213198011f, 4.061549402f), make_float2(0.2148494292f, 4.157871801f), make_float2(0.2167420483f, 4.253445104f),
              make_float2(0.2199813776f, 4.344835941f), make_float2(0.2234161891f, 4.435844004f), make_float2(0.2295246247f, 4.521616847f), make_float2(0.2356330603f, 4.607389689f),
              make_float2(0.2414331723f, 4.691710462f), make_float2(0.2471822878f, 4.775791063f), make_float2(0.2508542723f, 4.86128937f), make_float2(0.252529685f, 4.948150398f),
              make_float2(0.2542102449f, 5.034989516f), make_float2(0.2563900276f, 5.119703628f), make_float2(0.2585698103f, 5.204417741f), make_float2(0.2624130423f, 5.287222134f)};

std::array CuZn{make_float2(1.503f, 1.815f), make_float2(1.497f, 1.818f), make_float2(1.487f, 1.818f), make_float2(1.471f, 1.813f),
                make_float2(1.445f, 1.805f), make_float2(1.405f, 1.794f), make_float2(1.35f, 1.786f), make_float2(1.278f, 1.784f),
                make_float2(1.191f, 1.797f), make_float2(1.094f, 1.829f), make_float2(0.994f, 1.883f), make_float2(0.9f, 1.957f),
                make_float2(0.816f, 2.046f), make_float2(0.745f, 2.145f), make_float2(0.686f, 2.25f), make_float2(0.639f, 2.358f),
                make_float2(0.602f, 2.464f), make_float2(0.573f, 2.568f), make_float2(0.549f, 2.668f), make_float2(0.527f, 2.765f),
                make_float2(0.505f, 2.86f), make_float2(0.484f, 2.958f), make_float2(0.468f, 3.059f), make_float2(0.46f, 3.159f),
                make_float2(0.45f, 3.253f), make_float2(0.452f, 3.345f), make_float2(0.449f, 3.434f), make_float2(0.445f, 3.522f),
                make_float2(0.444f, 3.609f), make_float2(0.444f, 3.695f), make_float2(0.445f, 3.778f), make_float2(0.444f, 3.86f),
                make_float2(0.444f, 3.943f), make_float2(0.445f, 4.025f), make_float2(0.446f, 4.106f), make_float2(0.448f, 4.186f),
                make_float2(0.45f, 4.266f), make_float2(0.452f, 4.346f), make_float2(0.455f, 4.424f), make_float2(0.457f, 4.501f),
                make_float2(0.458f, 4.579f), make_float2(0.46f, 4.657f), make_float2(0.464f, 4.737f), make_float2(0.469f, 4.814f),
                make_float2(0.473f, 4.89f), make_float2(0.478f, 4.965f), make_float2(0.481f, 5.039f), make_float2(0.483f, 5.115f)};

std::array Fe{make_float2(1.968571429f, 2.384285714f), make_float2(2.035384615f, 2.440769231f), make_float2(2.112307692f, 2.494615385f), make_float2(2.1875f, 2.545f),
              make_float2(2.260625f, 2.593125f), make_float2(2.329375f, 2.636875f), make_float2(2.400555556f, 2.673333333f), make_float2(2.472777778f, 2.706666667f),
              make_float2(2.5295f, 2.737f), make_float2(2.5845f, 2.767f), make_float2(2.626f, 2.7925f), make_float2(2.666f, 2.8175f), make_float2(2.6952f, 2.8416f),
              make_float2(2.7232f, 2.8656f), make_float2(2.7592f, 2.8848f), make_float2(2.8072f, 2.8968f), make_float2(2.8552f, 2.9088f),
              make_float2(2.888928571f, 2.916428571f), make_float2(2.921071429f, 2.923571429f), make_float2(2.94969697f, 2.931818182f), make_float2(2.946666667f, 2.95f),
              make_float2(2.943636364f, 2.968181818f), make_float2(2.940606061f, 2.986363636f), make_float2(2.926285714f, 3.003714286f), make_float2(2.909142857f, 3.020857143f),
              make_float2(2.892f, 3.038f), make_float2(2.882857143f, 3.053571429f), make_float2(2.892380952f, 3.06547619f), make_float2(2.901904762f, 3.077380952f),
              make_float2(2.911428571f, 3.089285714f), make_float2(2.918666667f, 3.102f), make_float2(2.905333333f, 3.122f), make_float2(2.892f, 3.142f),
              make_float2(2.878666667f, 3.162f), make_float2(2.865333333f, 3.182f), make_float2(2.861153846f, 3.200384615f), make_float2(2.863076923f, 3.217692308f),
              make_float2(2.865f, 3.235f), make_float2(2.866923077f, 3.252307692f), make_float2(2.868846154f, 3.269615385f), make_float2(2.874307692f, 3.286769231f),
              make_float2(2.885076923f, 3.303692308f), make_float2(2.895846154f, 3.320615385f), make_float2(2.906615385f, 3.337538462f), make_float2(2.917384615f, 3.354461538f),
              make_float2(2.928153846f, 3.371384615f), make_float2(2.938923077f, 3.388307692f), make_float2(2.942535211f, 3.411549296f)};

std::array Ti{make_float2(1.854285714f, 2.882857143f), make_float2(1.913846154f, 2.904615385f), make_float2(1.983076923f, 2.927692308f), make_float2(2.040625f, 2.94125f),
              make_float2(2.09125f, 2.955625f), make_float2(2.12875f, 2.974375f), make_float2(2.167222222f, 2.991666667f), make_float2(2.206111111f, 3.008333333f),
              make_float2(2.237f, 3.0235f), make_float2(2.267f, 3.0385f), make_float2(2.2925f, 3.067f), make_float2(2.3175f, 3.097f),
              make_float2(2.3344f, 3.1324f), make_float2(2.3504f, 3.1684f), make_float2(2.3728f, 3.2076f), make_float2(2.4048f, 3.2516f),
              make_float2(2.4368f, 3.2956f), make_float2(2.472142857f, 3.341785714f), make_float2(2.507857143f, 3.388214286f), make_float2(2.541818182f, 3.434545455f),
              make_float2(2.56f, 3.48f), make_float2(2.578181818f, 3.525454545f), make_float2(2.596363636f, 3.570909091f), make_float2(2.616f, 3.612f),
              make_float2(2.636f, 3.652f), make_float2(2.656f, 3.692f), make_float2(2.676428571f, 3.728571429f), make_float2(2.697857143f, 3.757142857f),
              make_float2(2.719285714f, 3.785714286f), make_float2(2.740714286f, 3.814285714f), make_float2(2.762222222f, 3.842666667f), make_float2(2.784444444f, 3.869333333f),
              make_float2(2.806666667f, 3.896f), make_float2(2.828888889f, 3.922666667f), make_float2(2.851111111f, 3.949333333f), make_float2(2.876153846f, 3.965769231f),
              make_float2(2.903076923f, 3.975384615f), make_float2(2.93f, 3.985f), make_float2(2.956923077f, 3.994615385f), make_float2(2.983846154f, 4.004230769f),
              make_float2(3.012923077f, 4.01f), make_float2(3.045230769f, 4.01f), make_float2(3.077538462f, 4.01f), make_float2(3.109846154f, 4.01f),
              make_float2(3.142153846f, 4.01f), make_float2(3.174461538f, 4.01f), make_float2(3.206769231f, 4.01f), make_float2(3.220140845f, 4.003661972f)};

std::array V{make_float2(2.582857143f, 3.365714286f), make_float2(2.709230769f, 3.407692308f), make_float2(2.855384615f, 3.446153846f), make_float2(2.9825f, 3.466875f),
             make_float2(3.115f, 3.481875f), make_float2(3.265f, 3.488125f), make_float2(3.391666667f, 3.49f), make_float2(3.508333333f, 3.49f),
             make_float2(3.5515f, 3.4765f), make_float2(3.5865f, 3.4615f), make_float2(3.689f, 3.424f), make_float2(3.799f, 3.384f),
             make_float2(3.8496f, 3.3368f), make_float2(3.8936f, 3.2888f), make_float2(3.9104f, 3.2472f), make_float2(3.8864f, 3.2152f),
             make_float2(3.8624f, 3.1832f), make_float2(3.905f, 3.135f), make_float2(3.955f, 3.085f), make_float2(3.994848485f, 3.038787879f),
             make_float2(3.943333333f, 3.026666667f), make_float2(3.891818182f, 3.014545455f), make_float2(3.84030303f, 3.002424242f), make_float2(3.763714286f, 3.004571429f),
             make_float2(3.680857143f, 3.010285714f), make_float2(3.598f, 3.016f), make_float2(3.519285714f, 3.025f), make_float2(3.450238095f, 3.041666667f),
             make_float2(3.381190476f, 3.058333333f), make_float2(3.312142857f, 3.075f), make_float2(3.248444444f, 3.091333333f), make_float2(3.232888889f, 3.104666667f),
             make_float2(3.217333333f, 3.118f), make_float2(3.201777778f, 3.131333333f), make_float2(3.186222222f, 3.144666667f), make_float2(3.182307692f, 3.155769231f),
             make_float2(3.186153846f, 3.165384615f), make_float2(3.19f, 3.175f), make_float2(3.193846154f, 3.184615385f), make_float2(3.197692308f, 3.194230769f),
             make_float2(3.197538462f, 3.203076923f), make_float2(3.191384615f, 3.210769231f), make_float2(3.185230769f, 3.218461538f), make_float2(3.179076923f, 3.226153846f),
             make_float2(3.172923077f, 3.233846154f), make_float2(3.166769231f, 3.241538462f), make_float2(3.160615385f, 3.249230769f), make_float2(3.154929577f, 3.261408451f)};

std::array VN{make_float2(2.175093063f, 1.59177665f), make_float2(2.166632826f, 1.612081218f), make_float2(2.158172589f, 1.632385787f), make_float2(2.149712352f, 1.652690355f),
              make_float2(2.141252115f, 1.672994924f), make_float2(2.132791878f, 1.693299492f), make_float2(2.133244552f, 1.722711864f), make_float2(2.138087167f, 1.756610169f),
              make_float2(2.142929782f, 1.790508475f), make_float2(2.147772397f, 1.82440678f), make_float2(2.152615012f, 1.858305085f), make_float2(2.157457627f, 1.89220339f),
              make_float2(2.162300242f, 1.926101695f), make_float2(2.167142857f, 1.96f), make_float2(2.175951613f, 1.996201613f), make_float2(2.190467742f, 2.035717742f),
              make_float2(2.204983871f, 2.075233871f), make_float2(2.2195f, 2.11475f), make_float2(2.234016129f, 2.154266129f), make_float2(2.248532258f, 2.193782258f),
              make_float2(2.263048387f, 2.233298387f), make_float2(2.277564516f, 2.272814516f), make_float2(2.292080645f, 2.312330645f), make_float2(2.306596774f, 2.351846774f),
              make_float2(2.321112903f, 2.391362903f), make_float2(2.335629032f, 2.430879032f), make_float2(2.350183841f, 2.470217707f), make_float2(2.368567973f, 2.491988389f),
              make_float2(2.386952104f, 2.513759071f), make_float2(2.405336236f, 2.535529753f), make_float2(2.423720368f, 2.557300435f), make_float2(2.442104499f, 2.579071118f),
              make_float2(2.460488631f, 2.6008418f), make_float2(2.478872762f, 2.622612482f), make_float2(2.497256894f, 2.644383164f), make_float2(2.515641026f, 2.666153846f),
              make_float2(2.534025157f, 2.687924528f), make_float2(2.552409289f, 2.70969521f), make_float2(2.57079342f, 2.731465893f), make_float2(2.589177552f, 2.753236575f),
              make_float2(2.607561684f, 2.775007257f), make_float2(2.625945815f, 2.796777939f), make_float2(2.644329947f, 2.818548621f), make_float2(2.662714078f, 2.840319303f),
              make_float2(2.68109821f, 2.862089985f), make_float2(2.699482342f, 2.883860668f), make_float2(2.717866473f, 2.90563135f), make_float2(2.733784176f, 2.926087588f)};

std::array Li{make_float2(0.3093694896f, 1.551618053f), make_float2(0.2954704779f, 1.607933493f), make_float2(0.2839774394f, 1.659737734f), make_float2(0.2723517073f, 1.707465366f),
              make_float2(0.2625129268f, 1.752152805f), make_float2(0.2495064254f, 1.813613796f), make_float2(0.237044465f, 1.857966876f), make_float2(0.2257989067f, 1.919315307f),
              make_float2(0.2168825225f, 1.96834955f), make_float2(0.2071535404f, 2.030163478f), make_float2(0.1962138337f, 2.087298868f), make_float2(0.1886596222f, 2.140744f),
              make_float2(0.1805051444f, 2.198260759f), make_float2(0.1717437616f, 2.259857472f), make_float2(0.1656800198f, 2.312320138f), make_float2(0.1593099241f, 2.374155806f),
              make_float2(0.1535770792f, 2.432204795f), make_float2(0.148790401f, 2.486881665f), make_float2(0.1458201169f, 2.540118815f), make_float2(0.1438000835f, 2.604893439f),
              make_float2(0.1440838019f, 2.652322141f), make_float2(0.1450512281f, 2.704840877f), make_float2(0.1460149235f, 2.764252629f), make_float2(0.1465101894f, 2.82960587f),
              make_float2(0.1476233333f, 2.87878f), make_float2(0.1484485714f, 2.931272169f), make_float2(0.1489886038f, 2.98749756f), make_float2(0.1500640755f, 3.049692528f),
              make_float2(0.15112474f, 3.109828183f), make_float2(0.1521647878f, 3.167096563f), make_float2(0.1538287238f, 3.221037521f), make_float2(0.1557062053f, 3.273839563f),
              make_float2(0.1570795704f, 3.330329431f), make_float2(0.1584818293f, 3.387203049f), make_float2(0.1601637398f, 3.446395122f), make_float2(0.1617818234f, 3.505818676f),
              make_float2(0.1630438196f, 3.566533647f), make_float2(0.1643836923f, 3.626885538f), make_float2(0.1667909321f, 3.682261104f), make_float2(0.1691981719f, 3.73763667f),
              make_float2(0.170738569f, 3.793493918f), make_float2(0.1721355026f, 3.849430886f), make_float2(0.17413376f, 3.90532656f), make_float2(0.17670976f, 3.96118256f),
              make_float2(0.1792768042f, 4.017031013f), make_float2(0.1809572243f, 4.072132288f), make_float2(0.1826376444f, 4.127233563f), make_float2(0.1844895579f, 4.181916168f)};

}// namespace ior

using namespace luisa::compute;

class MetalSurface final : public Surface {

private:
    const Texture *_roughness;
    const Texture *_kd;
    luisa::unique_ptr<Constant<float2>> _eta;
    bool _remap_roughness;

public:
    MetalSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _roughness{scene->load_texture(desc->property_node_or_default("roughness"))},
          _kd{scene->load_texture(desc->property_node_or_default("Kd"))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", true)} {
        if (auto eta_name = desc->property_string_or_default("eta"); !eta_name.empty()) {
            for (auto &c : eta_name) { c = static_cast<char>(tolower(c)); }
            if (eta_name == "ag" || eta_name == "silver") {
                _eta = luisa::make_unique<Constant<float2>>(ior::Ag);
            } else if (eta_name == "al" || eta_name == "aluminium") {
                _eta = luisa::make_unique<Constant<float2>>(ior::Al);
            } else if (eta_name == "au" || eta_name == "gold") {
                _eta = luisa::make_unique<Constant<float2>>(ior::Au);
            } else if (eta_name == "cu" || eta_name == "copper") {
                _eta = luisa::make_unique<Constant<float2>>(ior::Cu);
            } else if (eta_name == "cuzn" || eta_name == "cu-zn" || eta_name == "brass") {
                _eta = luisa::make_unique<Constant<float2>>(ior::CuZn);
            } else if (eta_name == "fe" || eta_name == "iron") {
                _eta = luisa::make_unique<Constant<float2>>(ior::Fe);
            } else if (eta_name == "ti" || eta_name == "titanium") {
                _eta = luisa::make_unique<Constant<float2>>(ior::Ti);
            } else if (eta_name == "v" || eta_name == "vanadium") {
                _eta = luisa::make_unique<Constant<float2>>(ior::V);
            } else if (eta_name == "vn") {
                _eta = luisa::make_unique<Constant<float2>>(ior::VN);
            } else if (eta_name == "li" || eta_name == "lithium") {
                _eta = luisa::make_unique<Constant<float2>>(ior::Li);
            } else [[unlikely]] {
                LUISA_WARNING_WITH_LOCATION(
                    "Unknown metal '{}'. "
                    "Fallback to Aluminium. [{}]",
                    eta_name,
                    desc->source_location().string());
                _eta = luisa::make_unique<Constant<float2>>(ior::Al);
            }
        } else {
            auto eta = desc->property_float_list("eta");
            if (eta.size() % 3u != 0u) [[unlikely]] {
                LUISA_ERROR(
                    "Invalid eta list size: {}. [{}]",
                    eta.size(), desc->source_location().string());
            }
            luisa::vector<float> lambda(eta.size() / 3u);
            luisa::vector<float> n(eta.size() / 3u);
            luisa::vector<float> k(eta.size() / 3u);
            for (auto i = 0u; i < eta.size() / 3u; i++) {
                lambda[i] = eta[i * 3u + 0u];
                n[i] = eta[i * 3u + 1u];
                k[i] = eta[i * 3u + 2u];
            }
            if (!std::is_sorted(lambda.cbegin(), lambda.cend())) [[unlikely]] {
                LUISA_ERROR(
                    "Unsorted wavelengths in eta list. [{}]",
                    desc->source_location().string());
            }
            if (lambda.front() > visible_wavelength_min ||
                lambda.back() < visible_wavelength_max) [[unlikely]] {
                LUISA_ERROR(
                    "Invalid wavelength range [{}, {}] in eta list. [{}]",
                    lambda.front(), lambda.back(), desc->source_location().string());
            }
            // TODO: scan rather than binary search
            luisa::vector<float2> lut(ior::lut_size);
            for (auto i = 0u; i < ior::lut_size; i++) {
                auto wavelength = static_cast<float>(
                    i * ior::lut_step + ior::lut_min);
                auto lb = std::lower_bound(
                    lambda.cbegin(), lambda.cend(), wavelength);
                auto index = std::clamp(
                    static_cast<size_t>(std::distance(lambda.cbegin(), lb)),
                    static_cast<size_t>(1u), lambda.size() - 1u);
                auto t = (wavelength - lambda[index - 1u]) /
                         (lambda[index] - lambda[index - 1u]);
                lut[i] = make_float2(
                    std::lerp(n[index - 1u], n[index], t),
                    std::lerp(k[index - 1u], k[index], t));
            }
            _eta = luisa::make_unique<Constant<float2>>(lut);
        }
        LUISA_RENDER_CHECK_GENERIC_TEXTURE(MetalSurface, roughness, 1);
        LUISA_RENDER_CHECK_ALBEDO_TEXTURE(MetalSurface, kd);
    }
    [[nodiscard]] auto &eta() const noexcept { return *_eta; }
    [[nodiscard]] auto remap_roughness() const noexcept { return _remap_roughness; }
    [[nodiscard]] string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }

private:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class MetalInstance final : public Surface::Instance {

private:
    const Texture::Instance *_roughness;
    const Texture::Instance *_kd;

public:
    MetalInstance(const Pipeline &pipeline, const Surface *surface,
                  const Texture::Instance *roughness, const Texture::Instance *Kd) noexcept
        : Surface::Instance{pipeline, surface},
          _roughness{roughness}, _kd{Kd} {}

private:
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> MetalSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto roughness = pipeline.build_texture(command_buffer, _roughness);
    auto Kd = pipeline.build_texture(command_buffer, _kd);
    return luisa::make_unique<MetalInstance>(pipeline, this, roughness, Kd);
}

class MetalClosure final : public Surface::Closure {

private:
    SampledSpectrum _eta_i;
    luisa::optional<SampledSpectrum> _refl;
    luisa::unique_ptr<FresnelConductor> _fresnel;
    luisa::unique_ptr<TrowbridgeReitzDistribution> _distrib;
    luisa::unique_ptr<MicrofacetReflection> _lobe;

public:
    MetalClosure(
        const Surface::Instance *instance, const Interaction &it, const SampledWavelengths &swl, Expr<float> time,
        const SampledSpectrum &n, const SampledSpectrum &k, luisa::optional<SampledSpectrum> refl, Expr<float2> alpha) noexcept
        : Surface::Closure{instance, it, swl, time}, _eta_i{swl.dimension(), 1.f}, _refl{std::move(refl)},
          _fresnel{luisa::make_unique<FresnelConductor>(_eta_i, n, k)},
          _distrib{luisa::make_unique<TrowbridgeReitzDistribution>(alpha)},
          _lobe{luisa::make_unique<MicrofacetReflection>(_eta_i, _distrib.get(), _fresnel.get())} {}

private:
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _it.wo_local();
        auto wi_local = _it.shading().world_to_local(wi);
        auto f = _lobe->evaluate(wo_local, wi_local);
        if (_refl) { f *= *_refl; }
        auto pdf = _lobe->pdf(wo_local, wi_local);
        return {.f = f * abs_cos_theta(wi_local),
                .pdf = pdf,
                .roughness = _distrib->alpha(),
                .eta = _eta_i};
    }
    [[nodiscard]] Surface::Sample sample(Expr<float>, Expr<float2> u) const noexcept override {
        auto wo_local = _it.wo_local();
        auto pdf = def(0.f);
        auto wi_local = def(make_float3(0.f, 0.f, 1.f));
        auto f = _lobe->sample(wo_local, &wi_local, u, &pdf);
        if (_refl) { f *= *_refl; }
        auto wi = _it.shading().local_to_world(wi_local);
        return {.wi = wi,
                .eval = {.f = f * abs_cos_theta(wi_local),
                         .pdf = pdf,
                         .roughness = _distrib->alpha(),
                         .eta = _eta_i}};
    }
    void backward(Expr<float3> wi, const SampledSpectrum &df) const noexcept override {
        // Metal surface is not differentiable
    }
};

luisa::unique_ptr<Surface::Closure> MetalInstance::closure(
    const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
    auto alpha = def(make_float2(.5f));
    if (_roughness != nullptr) {
        auto r = _roughness->evaluate(it, swl, time);
        auto remap = node<MetalSurface>()->remap_roughness();
        auto r2a = [](auto &&x) noexcept { return TrowbridgeReitzDistribution::roughness_to_alpha(x); };
        alpha = _roughness->node()->channels() == 1u ?
                    (remap ? make_float2(r2a(r.x)) : r.xx()) :
                    (remap ? r2a(r.xy()) : r.xy());
    }
    auto sample_eta_k = [&lut = node<MetalSurface>()->eta()](auto lambda) noexcept {
        auto lo = (lambda - visible_wavelength_min) / 5.f;
        auto il = cast<uint>(lo);
        auto ih = min(il + 1u, ior::lut_size - 1u);
        auto t = fract(lo);
        return lerp(lut.read(il), lut.read(ih), t);
    };
    SampledSpectrum eta{swl.dimension()};
    SampledSpectrum k{swl.dimension()};
    for (auto i = 0u; i < swl.dimension(); i++) {
        auto eta_k = sample_eta_k(swl.lambda(i));
        eta[i] = eta_k.x;
        k[i] = eta_k.y;
    }
    luisa::optional<SampledSpectrum> refl;
    if (_kd != nullptr) {
        refl.emplace(_kd->evaluate_albedo_spectrum(it, swl, time).value);
    }
    return luisa::make_unique<MetalClosure>(
        this, it, swl, time, eta, k, std::move(refl), alpha);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MetalSurface)
