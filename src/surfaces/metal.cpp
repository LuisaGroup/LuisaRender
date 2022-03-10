//
// Created by Mike Smith on 2022/1/9.
//

#include <util/sampling.h>
#include <util/scattering.h>
#include <base/surface.h>
#include <base/interaction.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

namespace ior {

static constexpr auto lut_step = 5u;
static constexpr auto lut_min = static_cast<uint>(visible_wavelength_min);
static constexpr auto lut_max = static_cast<uint>(visible_wavelength_max);
static constexpr auto lut_size = (lut_max - lut_min) / lut_step + 1;

static constexpr std::array Ag{
    make_float2(0.1937697969f, 1.542775784f), make_float2(0.1863794945f, 1.611995116f), make_float2(0.1919493526f, 1.641277522f),
    make_float2(0.1989955973f, 1.666440869f), make_float2(0.1985622426f, 1.718246893f), make_float2(0.194492584f, 1.779166452f),
    make_float2(0.1878545676f, 1.838499847f), make_float2(0.180016041f, 1.894456342f), make_float2(0.1729970816f, 1.950617723f),
    make_float2(0.1727111067f, 2.01114909f), make_float2(0.1728184204f, 2.070960395f), make_float2(0.1714506271f, 2.12871668f),
    make_float2(0.1669451194f, 2.183143955f), make_float2(0.1623365906f, 2.234233918f), make_float2(0.1595063655f, 2.282830594f),
    make_float2(0.1585064387f, 2.329077209f), make_float2(0.157540771f, 2.374538697f), make_float2(0.1553781177f, 2.421513092f),
    make_float2(0.1516909674f, 2.470420468f), make_float2(0.1475535859f, 2.519120932f), make_float2(0.1433830656f, 2.567380701f),
    make_float2(0.139524076f, 2.613547767f), make_float2(0.136052175f, 2.658984755f), make_float2(0.1330996471f, 2.703442166f),
    make_float2(0.1314127294f, 2.746217439f), make_float2(0.1304769255f, 2.787994397f), make_float2(0.1301526264f, 2.830014841f),
    make_float2(0.1300240817f, 2.872113229f), make_float2(0.1299751091f, 2.91763738f), make_float2(0.1299444802f, 2.963951269f),
    make_float2(0.1299611827f, 3.009739205f), make_float2(0.1299905863f, 3.055386004f), make_float2(0.1300194835f, 3.097640223f),
    make_float2(0.1300481421f, 3.138296806f), make_float2(0.1298398444f, 3.178399908f), make_float2(0.1293764383f, 3.21790713f),
    make_float2(0.1286377125f, 3.25768637f), make_float2(0.126707271f, 3.298643034f), make_float2(0.1247768295f, 3.339599698f),
    make_float2(0.1229918902f, 3.380331699f), make_float2(0.1212451994f, 3.421004641f), make_float2(0.1199247512f, 3.461363987f),
    make_float2(0.1196626603f, 3.500944672f), make_float2(0.1194005694f, 3.540525357f), make_float2(0.1197220763f, 3.57951429f),
    make_float2(0.1203364144f, 3.618206299f), make_float2(0.1209507524f, 3.656898309f), make_float2(0.122437242f, 3.69393488f),
    make_float2(0.1239997394f, 3.73082718f), make_float2(0.1255724448f, 3.76774829f), make_float2(0.1273912211f, 3.805363891f),
    make_float2(0.1292099974f, 3.842979492f), make_float2(0.1310255041f, 3.880650044f), make_float2(0.1326376071f, 3.921739011f),
    make_float2(0.1342497102f, 3.962827979f), make_float2(0.1358618132f, 4.003916947f), make_float2(0.1370939161f, 4.045474725f),
    make_float2(0.1382518772f, 4.087123972f), make_float2(0.1394098384f, 4.128773219f), make_float2(0.1400087579f, 4.169896413f),
    make_float2(0.1400266186f, 4.210472834f), make_float2(0.1400444793f, 4.251049256f), make_float2(0.14006234f, 4.291625677f),
    make_float2(0.1400467045f, 4.331115513f), make_float2(0.1400297838f, 4.370563658f), make_float2(0.140012863f, 4.410011804f),
    make_float2(0.1402627758f, 4.448910495f), make_float2(0.1413585553f, 4.486067405f), make_float2(0.1424543348f, 4.523224315f),
    make_float2(0.1435501144f, 4.560381225f), make_float2(0.1445817677f, 4.597457767f), make_float2(0.1454664645f, 4.634350128f),
    make_float2(0.1463511613f, 4.671242489f), make_float2(0.1472358582f, 4.70813485f), make_float2(0.1479344888f, 4.745183096f),
    make_float2(0.1474537322f, 4.783219419f), make_float2(0.1469729757f, 4.821255743f), make_float2(0.1464922192f, 4.859292066f),
    make_float2(0.1460114627f, 4.89732839f), make_float2(0.1454365913f, 4.935859112f), make_float2(0.144824417f, 4.974585793f),
    make_float2(0.1442122427f, 5.013312474f), make_float2(0.1436000683f, 5.052039155f), make_float2(0.1429982714f, 5.090786172f),
    make_float2(0.1429108603f, 5.130541219f), make_float2(0.1428234492f, 5.170296267f), make_float2(0.1427360381f, 5.210051314f),
    make_float2(0.142648627f, 5.249806361f), make_float2(0.1425723274f, 5.28955854f), make_float2(0.1430293242f, 5.329173053f),
    make_float2(0.1434863209f, 5.368787566f), make_float2(0.1439433177f, 5.408402079f), make_float2(0.1444003145f, 5.448016592f),
    make_float2(0.1448573113f, 5.487631105f), make_float2(0.1458370844f, 5.526618752f)};

static constexpr std::array Al{
    make_float2(0.3970816731f, 4.372694111f), make_float2(0.4077589264f, 4.433990231f), make_float2(0.418897724f, 4.492555044f),
    make_float2(0.4303707087f, 4.551616268f), make_float2(0.4421600534f, 4.614741667f), make_float2(0.4541161368f, 4.67865256f),
    make_float2(0.4660888111f, 4.740831653f), make_float2(0.4780633756f, 4.801045767f), make_float2(0.4901259804f, 4.860607995f),
    make_float2(0.5024709476f, 4.920186112f), make_float2(0.5148594252f, 4.98035513f), make_float2(0.5271883634f, 5.041784988f),
    make_float2(0.5393679384f, 5.105134767f), make_float2(0.5517333395f, 5.168119932f), make_float2(0.5643557083f, 5.229618068f),
    make_float2(0.5772300992f, 5.289613676f), make_float2(0.5905420465f, 5.347553738f), make_float2(0.6040955128f, 5.407318213f),
    make_float2(0.6179529129f, 5.469422816f), make_float2(0.6320824311f, 5.529687301f), make_float2(0.6462718721f, 5.590148464f),
    make_float2(0.660482672f, 5.65362818f), make_float2(0.6748840364f, 5.716104038f), make_float2(0.6895410892f, 5.777232979f),
    make_float2(0.7046480176f, 5.83859207f), make_float2(0.720021938f, 5.900087752f), make_float2(0.7359216949f, 5.959791837f),
    make_float2(0.7519897821f, 6.018922398f), make_float2(0.7686498231f, 6.078175846f), make_float2(0.7854463268f, 6.137457624f),
    make_float2(0.8028348113f, 6.198608797f), make_float2(0.8203821522f, 6.260261616f), make_float2(0.8386797108f, 6.323296574f),
    make_float2(0.8573305628f, 6.38698241f), make_float2(0.8764468509f, 6.447290903f), make_float2(0.8960642301f, 6.503963331f),
    make_float2(0.9159949541f, 6.561749865f), make_float2(0.9372819846f, 6.624358778f), make_float2(0.9585690152f, 6.686967692f),
    make_float2(0.9810199444f, 6.747420906f), make_float2(1.00377683f, 6.807307448f), make_float2(1.026571944f, 6.86642986f),
    make_float2(1.049461979f, 6.923654947f), make_float2(1.072352014f, 6.980880034f), make_float2(1.09697412f, 7.036373115f),
    make_float2(1.122465327f, 7.090997129f), make_float2(1.147956533f, 7.145621142f), make_float2(1.172357097f, 7.201101937f),
    make_float2(1.196662613f, 7.2566574f), make_float2(1.221053742f, 7.312239202f), make_float2(1.247508671f, 7.368455925f),
    make_float2(1.273963599f, 7.424672648f), make_float2(1.300447877f, 7.48084599f), make_float2(1.328757979f, 7.534320627f),
    make_float2(1.357068081f, 7.587795265f), make_float2(1.385378183f, 7.641269902f), make_float2(1.415004287f, 7.692507287f),
    make_float2(1.444887155f, 7.743308163f), make_float2(1.474770023f, 7.794109039f), make_float2(1.505291566f, 7.846412705f),
    make_float2(1.536476938f, 7.900278348f), make_float2(1.56766231f, 7.95414399f), make_float2(1.598847682f, 8.008009632f),
    make_float2(1.636212234f, 8.061731762f), make_float2(1.67381387f, 8.115448386f), make_float2(1.711415507f, 8.169165009f),
    make_float2(1.750357498f, 8.220966763f), make_float2(1.793548436f, 8.266698344f), make_float2(1.836739375f, 8.312429926f),
    make_float2(1.879930313f, 8.358161508f), make_float2(1.92676609f, 8.403121288f), make_float2(1.981954663f, 8.446312345f),
    make_float2(2.037143236f, 8.489503402f), make_float2(2.092331808f, 8.532694459f), make_float2(2.148323782f, 8.571541441f),
    make_float2(2.209408138f, 8.582853359f), make_float2(2.270492494f, 8.594165277f), make_float2(2.33157685f, 8.605477194f),
    make_float2(2.392661206f, 8.616789112f), make_float2(2.443547792f, 8.616950201f), make_float2(2.490392437f, 8.612691597f),
    make_float2(2.537237082f, 8.608432993f), make_float2(2.584081727f, 8.604174388f), make_float2(2.630672449f, 8.599406663f),
    make_float2(2.664676766f, 8.569402853f), make_float2(2.698681083f, 8.539399044f), make_float2(2.732685401f, 8.509395235f),
    make_float2(2.766689718f, 8.479391426f), make_float2(2.799770355f, 8.449464163f), make_float2(2.788518896f, 8.423210758f),
    make_float2(2.777267437f, 8.396957353f), make_float2(2.766015978f, 8.370703948f), make_float2(2.754764518f, 8.344450542f),
    make_float2(2.743513059f, 8.318197137f), make_float2(2.720695662f, 8.297934788f)};

static constexpr std::array Au{
    make_float2(1.726248938f, 1.85351432f), make_float2(1.715366257f, 1.863314429f), make_float2(1.706064787f, 1.882606367f),
    make_float2(1.697249651f, 1.903089421f), make_float2(1.687649576f, 1.918247288f), make_float2(1.678419475f, 1.930449961f),
    make_float2(1.670683654f, 1.940870883f), make_float2(1.664236481f, 1.949568629f), make_float2(1.65791634f, 1.956026265f),
    make_float2(1.649718391f, 1.95860004f), make_float2(1.641447387f, 1.958665792f), make_float2(1.634036283f, 1.956373797f),
    make_float2(1.62832588f, 1.951644869f), make_float2(1.620192447f, 1.943974091f), make_float2(1.60950048f, 1.934899111f),
    make_float2(1.596334777f, 1.924566534f), make_float2(1.574347605f, 1.911390537f), make_float2(1.54564322f, 1.896316753f),
    make_float2(1.508458089f, 1.878849833f), make_float2(1.464300542f, 1.861030919f), make_float2(1.418607767f, 1.843105381f),
    make_float2(1.372368565f, 1.824999235f), make_float2(1.321224482f, 1.810205112f), make_float2(1.263499312f, 1.799854871f),
    make_float2(1.189900705f, 1.796461427f), make_float2(1.106881535f, 1.797196701f), make_float2(1.020243859f, 1.813977192f),
    make_float2(0.9324478308f, 1.835894063f), make_float2(0.8511633401f, 1.886770717f), make_float2(0.7713799478f, 1.944323443f),
    make_float2(0.6997216254f, 2.01763491f), make_float2(0.6302436408f, 2.095175186f), make_float2(0.5720483809f, 2.183785131f),
    make_float2(0.5191663921f, 2.277608014f), make_float2(0.4729087806f, 2.371122542f), make_float2(0.4337830078f, 2.464305093f),
    make_float2(0.3975742176f, 2.55493212f), make_float2(0.3739915268f, 2.634497614f), make_float2(0.3504088361f, 2.714063108f),
    make_float2(0.3330124947f, 2.777848319f), make_float2(0.3172423721f, 2.837485333f), make_float2(0.3017966941f, 2.886640402f),
    make_float2(0.2871566093f, 2.909768875f), make_float2(0.2725165246f, 2.932897347f), make_float2(0.259956335f, 2.947748068f),
    make_float2(0.2484397721f, 2.958445271f), make_float2(0.2369232091f, 2.969142474f), make_float2(0.2284544796f, 2.98437242f),
    make_float2(0.2202513683f, 2.999997394f), make_float2(0.2121329686f, 3.01559273f), make_float2(0.2060566022f, 3.030473627f),
    make_float2(0.1999802358f, 3.045354524f), make_float2(0.1939191333f, 3.060049764f), make_float2(0.1888075871f, 3.063195331f),
    make_float2(0.1836960409f, 3.066340898f), make_float2(0.1785844947f, 3.069486465f), make_float2(0.1748119963f, 3.090003429f),
    make_float2(0.1713007593f, 3.113909724f), make_float2(0.1677895223f, 3.137816018f), make_float2(0.1653049288f, 3.19112208f),
    make_float2(0.1638874119f, 3.274985782f), make_float2(0.162469895f, 3.358849484f), make_float2(0.1610523781f, 3.442713186f),
    make_float2(0.1607413412f, 3.537425847f), make_float2(0.1604727581f, 3.632554764f), make_float2(0.160204175f, 3.727683682f),
    make_float2(0.1600533107f, 3.81752775f), make_float2(0.160275617f, 3.890618759f), make_float2(0.1604979233f, 3.963709768f),
    make_float2(0.1607202296f, 4.036800777f), make_float2(0.1611028001f, 4.107318852f), make_float2(0.1618526449f, 4.171940592f),
    make_float2(0.1626024896f, 4.236562331f), make_float2(0.1633523344f, 4.301184071f), make_float2(0.1641695585f, 4.364805488f),
    make_float2(0.1654138695f, 4.422086324f), make_float2(0.1666581804f, 4.479367161f), make_float2(0.1679024914f, 4.536647998f),
    make_float2(0.1691468023f, 4.593928835f), make_float2(0.1704911848f, 4.648281823f), make_float2(0.1718752311f, 4.701474343f),
    make_float2(0.1732592774f, 4.754666863f), make_float2(0.1746433237f, 4.807859383f), make_float2(0.1760212613f, 4.861051445f),
    make_float2(0.1770963978f, 4.914220796f), make_float2(0.1781715343f, 4.967390146f), make_float2(0.1792466708f, 5.020559496f),
    make_float2(0.1803218073f, 5.073728847f), make_float2(0.1814003566f, 5.126824152f), make_float2(0.1826427052f, 5.17636564f),
    make_float2(0.1838850538f, 5.225907128f), make_float2(0.1851274025f, 5.275448616f), make_float2(0.1863697511f, 5.324990104f),
    make_float2(0.1876120997f, 5.374531592f), make_float2(0.1892216027f, 5.419107323f)};

static constexpr std::array Cu{
    make_float2(1.280194277f, 1.933855609f), make_float2(1.268689452f, 1.951404436f), make_float2(1.249454471f, 1.972017413f),
    make_float2(1.228044971f, 2.009613911f), make_float2(1.206443503f, 2.094096699f), make_float2(1.188222279f, 2.173641903f),
    make_float2(1.177668194f, 2.196398007f), make_float2(1.174512775f, 2.166785883f), make_float2(1.175019456f, 2.13023396f),
    make_float2(1.176925956f, 2.153160001f), make_float2(1.17878947f, 2.185819898f), make_float2(1.179539754f, 2.219803367f),
    make_float2(1.178201378f, 2.248311125f), make_float2(1.176091783f, 2.275352885f), make_float2(1.174279952f, 2.301176317f),
    make_float2(1.172821392f, 2.325841178f), make_float2(1.171013765f, 2.349021495f), make_float2(1.168719322f, 2.371732113f),
    make_float2(1.165807858f, 2.393856878f), make_float2(1.162672611f, 2.415201883f), make_float2(1.159567491f, 2.436338568f),
    make_float2(1.156862101f, 2.457221663f), make_float2(1.154051412f, 2.477448453f), make_float2(1.15109944f, 2.496794652f),
    make_float2(1.147587828f, 2.514696715f), make_float2(1.14374408f, 2.531741714f), make_float2(1.139769271f, 2.546829525f),
    make_float2(1.135752506f, 2.561290804f), make_float2(1.133619762f, 2.574287551f), make_float2(1.131921339f, 2.58694668f),
    make_float2(1.127238808f, 2.595373925f), make_float2(1.121755502f, 2.602665556f), make_float2(1.111475832f, 2.602081192f),
    make_float2(1.098937456f, 2.597787857f), make_float2(1.081428899f, 2.592924859f), make_float2(1.058569421f, 2.587448559f),
    make_float2(1.032979456f, 2.582490839f), make_float2(0.99557063f, 2.579777786f), make_float2(0.9581618042f, 2.577064733f),
    make_float2(0.9110551652f, 2.583613136f), make_float2(0.8613992414f, 2.592596117f), make_float2(0.806270476f, 2.607659851f),
    make_float2(0.7375526842f, 2.63782205f), make_float2(0.6688348924f, 2.667984248f), make_float2(0.602742548f, 2.709812819f),
    make_float2(0.5379675717f, 2.757495213f), make_float2(0.4731925953f, 2.805177608f), make_float2(0.4307115553f, 2.873436349f),
    make_float2(0.3901734284f, 2.943488315f), make_float2(0.3502061367f, 3.01376963f), make_float2(0.3239992232f, 3.089579534f),
    make_float2(0.2977923097f, 3.165389439f), make_float2(0.2717950365f, 3.241085789f), make_float2(0.2588393901f, 3.309717969f),
    make_float2(0.2458837437f, 3.378350149f), make_float2(0.2329280974f, 3.44698233f), make_float2(0.2266090293f, 3.51114933f),
    make_float2(0.2215848227f, 3.574445129f), make_float2(0.216560616f, 3.637740929f), make_float2(0.2133396824f, 3.696847125f),
    make_float2(0.2119930413f, 3.751598715f), make_float2(0.2106464003f, 3.806350306f), make_float2(0.2092997592f, 3.861101896f),
    make_float2(0.2102199705f, 3.911461865f), make_float2(0.2112271572f, 3.961653335f), make_float2(0.2122343439f, 4.011844806f),
    make_float2(0.213198011f, 4.061549402f), make_float2(0.2140237201f, 4.109710601f), make_float2(0.2148494292f, 4.157871801f),
    make_float2(0.2156751383f, 4.206033f), make_float2(0.2167420483f, 4.253445104f), make_float2(0.2183617129f, 4.299140522f),
    make_float2(0.2199813776f, 4.344835941f), make_float2(0.2216010422f, 4.390531359f), make_float2(0.2234161891f, 4.435844004f),
    make_float2(0.2264704069f, 4.478730425f), make_float2(0.2295246247f, 4.521616847f), make_float2(0.2325788425f, 4.564503268f),
    make_float2(0.2356330603f, 4.607389689f), make_float2(0.2385586145f, 4.649670161f), make_float2(0.2414331723f, 4.691710462f),
    make_float2(0.24430773f, 4.733750763f), make_float2(0.2471822878f, 4.775791063f), make_float2(0.250016566f, 4.817858856f),
    make_float2(0.2508542723f, 4.86128937f), make_float2(0.2516919787f, 4.904719884f), make_float2(0.252529685f, 4.948150398f),
    make_float2(0.2533673914f, 4.991580912f), make_float2(0.2542102449f, 5.034989516f), make_float2(0.2553001363f, 5.077346572f),
    make_float2(0.2563900276f, 5.119703628f), make_float2(0.257479919f, 5.162060685f), make_float2(0.2585698103f, 5.204417741f),
    make_float2(0.2596597017f, 5.246774797f), make_float2(0.2624130423f, 5.287222134f)};

static constexpr std::array CuZn{
    make_float2(1.503f, 1.815f), make_float2(1.5f, 1.8165f), make_float2(1.497f, 1.818f), make_float2(1.492f, 1.818f),
    make_float2(1.487f, 1.818f), make_float2(1.479f, 1.8155f), make_float2(1.471f, 1.813f), make_float2(1.458f, 1.809f),
    make_float2(1.445f, 1.805f), make_float2(1.425f, 1.7995f), make_float2(1.405f, 1.794f), make_float2(1.3775f, 1.79f),
    make_float2(1.35f, 1.786f), make_float2(1.314f, 1.785f), make_float2(1.278f, 1.784f), make_float2(1.2345f, 1.7905f),
    make_float2(1.191f, 1.797f), make_float2(1.1425f, 1.813f), make_float2(1.094f, 1.829f), make_float2(1.044f, 1.856f),
    make_float2(0.994f, 1.883f), make_float2(0.947f, 1.92f), make_float2(0.9f, 1.957f), make_float2(0.858f, 2.0015f),
    make_float2(0.816f, 2.046f), make_float2(0.7805f, 2.0955f), make_float2(0.745f, 2.145f), make_float2(0.7155f, 2.1975f),
    make_float2(0.686f, 2.25f), make_float2(0.6625f, 2.304f), make_float2(0.639f, 2.358f), make_float2(0.6205f, 2.411f),
    make_float2(0.602f, 2.464f), make_float2(0.5875f, 2.516f), make_float2(0.573f, 2.568f), make_float2(0.561f, 2.618f),
    make_float2(0.549f, 2.668f), make_float2(0.538f, 2.7165f), make_float2(0.527f, 2.765f), make_float2(0.516f, 2.8125f),
    make_float2(0.505f, 2.86f), make_float2(0.4945f, 2.909f), make_float2(0.484f, 2.958f), make_float2(0.476f, 3.0085f),
    make_float2(0.468f, 3.059f), make_float2(0.464f, 3.109f), make_float2(0.46f, 3.159f), make_float2(0.455f, 3.206f),
    make_float2(0.45f, 3.253f), make_float2(0.451f, 3.299f), make_float2(0.452f, 3.345f), make_float2(0.4505f, 3.3895f),
    make_float2(0.449f, 3.434f), make_float2(0.447f, 3.478f), make_float2(0.445f, 3.522f), make_float2(0.4445f, 3.5655f),
    make_float2(0.444f, 3.609f), make_float2(0.444f, 3.652f), make_float2(0.444f, 3.695f), make_float2(0.4445f, 3.7365f),
    make_float2(0.445f, 3.778f), make_float2(0.4445f, 3.819f), make_float2(0.444f, 3.86f), make_float2(0.444f, 3.9015f),
    make_float2(0.444f, 3.943f), make_float2(0.4445f, 3.984f), make_float2(0.445f, 4.025f), make_float2(0.4455f, 4.0655f),
    make_float2(0.446f, 4.106f), make_float2(0.447f, 4.146f), make_float2(0.448f, 4.186f), make_float2(0.449f, 4.226f),
    make_float2(0.45f, 4.266f), make_float2(0.451f, 4.306f), make_float2(0.452f, 4.346f), make_float2(0.4535f, 4.385f),
    make_float2(0.455f, 4.424f), make_float2(0.456f, 4.4625f), make_float2(0.457f, 4.501f), make_float2(0.4575f, 4.54f),
    make_float2(0.458f, 4.579f), make_float2(0.459f, 4.618f), make_float2(0.46f, 4.657f), make_float2(0.462f, 4.697f),
    make_float2(0.464f, 4.737f), make_float2(0.4665f, 4.7755f), make_float2(0.469f, 4.814f), make_float2(0.471f, 4.852f),
    make_float2(0.473f, 4.89f), make_float2(0.4755f, 4.9275f), make_float2(0.478f, 4.965f), make_float2(0.4795f, 5.002f),
    make_float2(0.481f, 5.039f), make_float2(0.482f, 5.077f), make_float2(0.483f, 5.115f)};

static constexpr std::array Fe{
    make_float2(1.968571429f, 2.384285714f), make_float2(2.000714286f, 2.412857143f), make_float2(2.035384615f, 2.440769231f),
    make_float2(2.073846154f, 2.467692308f), make_float2(2.112307692f, 2.494615385f), make_float2(2.15f, 2.52f),
    make_float2(2.1875f, 2.545f), make_float2(2.225f, 2.57f), make_float2(2.260625f, 2.593125f), make_float2(2.295f, 2.615f),
    make_float2(2.329375f, 2.636875f), make_float2(2.364444444f, 2.656666667f), make_float2(2.400555556f, 2.673333333f),
    make_float2(2.436666667f, 2.69f), make_float2(2.472777778f, 2.706666667f), make_float2(2.502f, 2.722f),
    make_float2(2.5295f, 2.737f), make_float2(2.557f, 2.752f), make_float2(2.5845f, 2.767f), make_float2(2.606f, 2.78f),
    make_float2(2.626f, 2.7925f), make_float2(2.646f, 2.805f), make_float2(2.666f, 2.8175f), make_float2(2.6812f, 2.8296f),
    make_float2(2.6952f, 2.8416f), make_float2(2.7092f, 2.8536f), make_float2(2.7232f, 2.8656f), make_float2(2.7372f, 2.8776f),
    make_float2(2.7592f, 2.8848f), make_float2(2.7832f, 2.8908f), make_float2(2.8072f, 2.8968f), make_float2(2.8312f, 2.9028f),
    make_float2(2.8552f, 2.9088f), make_float2(2.872857143f, 2.912857143f), make_float2(2.888928571f, 2.916428571f),
    make_float2(2.905f, 2.92f), make_float2(2.921071429f, 2.923571429f), make_float2(2.937142857f, 2.927142857f),
    make_float2(2.94969697f, 2.931818182f), make_float2(2.948181818f, 2.940909091f), make_float2(2.946666667f, 2.95f),
    make_float2(2.945151515f, 2.959090909f), make_float2(2.943636364f, 2.968181818f), make_float2(2.942121212f, 2.977272727f),
    make_float2(2.940606061f, 2.986363636f), make_float2(2.934857143f, 2.995142857f), make_float2(2.926285714f, 3.003714286f),
    make_float2(2.917714286f, 3.012285714f), make_float2(2.909142857f, 3.020857143f), make_float2(2.900571429f, 3.029428571f),
    make_float2(2.892f, 3.038f), make_float2(2.883428571f, 3.046571429f), make_float2(2.882857143f, 3.053571429f),
    make_float2(2.887619048f, 3.05952381f), make_float2(2.892380952f, 3.06547619f), make_float2(2.897142857f, 3.071428571f),
    make_float2(2.901904762f, 3.077380952f), make_float2(2.906666667f, 3.083333333f), make_float2(2.911428571f, 3.089285714f),
    make_float2(2.916190476f, 3.095238095f), make_float2(2.918666667f, 3.102f), make_float2(2.912f, 3.112f),
    make_float2(2.905333333f, 3.122f), make_float2(2.898666667f, 3.132f), make_float2(2.892f, 3.142f), make_float2(2.885333333f, 3.152f),
    make_float2(2.878666667f, 3.162f), make_float2(2.872f, 3.172f), make_float2(2.865333333f, 3.182f), make_float2(2.860192308f, 3.191730769f),
    make_float2(2.861153846f, 3.200384615f), make_float2(2.862115385f, 3.209038462f), make_float2(2.863076923f, 3.217692308f),
    make_float2(2.864038462f, 3.226346154f), make_float2(2.865f, 3.235f), make_float2(2.865961538f, 3.243653846f),
    make_float2(2.866923077f, 3.252307692f), make_float2(2.867884615f, 3.260961538f), make_float2(2.868846154f, 3.269615385f),
    make_float2(2.869807692f, 3.278269231f), make_float2(2.874307692f, 3.286769231f), make_float2(2.879692308f, 3.295230769f),
    make_float2(2.885076923f, 3.303692308f), make_float2(2.890461538f, 3.312153846f), make_float2(2.895846154f, 3.320615385f),
    make_float2(2.901230769f, 3.329076923f), make_float2(2.906615385f, 3.337538462f), make_float2(2.912f, 3.346f),
    make_float2(2.917384615f, 3.354461538f), make_float2(2.922769231f, 3.362923077f), make_float2(2.928153846f, 3.371384615f),
    make_float2(2.933538462f, 3.379846154f), make_float2(2.938923077f, 3.388307692f), make_float2(2.941126761f, 3.399577465f),
    make_float2(2.942535211f, 3.411549296f)};

static constexpr std::array Ti{
    make_float2(1.854285714f, 2.882857143f), make_float2(1.882857143f, 2.893571429f), make_float2(1.913846154f, 2.904615385f),
    make_float2(1.948461538f, 2.916153846f), make_float2(1.983076923f, 2.927692308f), make_float2(2.0125f, 2.935f),
    make_float2(2.040625f, 2.94125f), make_float2(2.06875f, 2.9475f), make_float2(2.09125f, 2.955625f), make_float2(2.11f, 2.965f),
    make_float2(2.12875f, 2.974375f), make_float2(2.147777778f, 2.983333333f), make_float2(2.167222222f, 2.991666667f),
    make_float2(2.186666667f, 3.0f), make_float2(2.206111111f, 3.008333333f), make_float2(2.222f, 3.016f), make_float2(2.237f, 3.0235f),
    make_float2(2.252f, 3.031f), make_float2(2.267f, 3.0385f), make_float2(2.28f, 3.052f), make_float2(2.2925f, 3.067f),
    make_float2(2.305f, 3.082f), make_float2(2.3175f, 3.097f), make_float2(2.3264f, 3.1144f), make_float2(2.3344f, 3.1324f),
    make_float2(2.3424f, 3.1504f), make_float2(2.3504f, 3.1684f), make_float2(2.3584f, 3.1864f), make_float2(2.3728f, 3.2076f),
    make_float2(2.3888f, 3.2296f), make_float2(2.4048f, 3.2516f), make_float2(2.4208f, 3.2736f), make_float2(2.4368f, 3.2956f),
    make_float2(2.454285714f, 3.318571429f), make_float2(2.472142857f, 3.341785714f), make_float2(2.49f, 3.365f),
    make_float2(2.507857143f, 3.388214286f), make_float2(2.525714286f, 3.411428571f), make_float2(2.541818182f, 3.434545455f),
    make_float2(2.550909091f, 3.457272727f), make_float2(2.56f, 3.48f), make_float2(2.569090909f, 3.502727273f),
    make_float2(2.578181818f, 3.525454545f), make_float2(2.587272727f, 3.548181818f), make_float2(2.596363636f, 3.570909091f),
    make_float2(2.606f, 3.592f), make_float2(2.616f, 3.612f), make_float2(2.626f, 3.632f), make_float2(2.636f, 3.652f),
    make_float2(2.646f, 3.672f), make_float2(2.656f, 3.692f), make_float2(2.666f, 3.712f), make_float2(2.676428571f, 3.728571429f),
    make_float2(2.687142857f, 3.742857143f), make_float2(2.697857143f, 3.757142857f), make_float2(2.708571429f, 3.771428571f),
    make_float2(2.719285714f, 3.785714286f), make_float2(2.73f, 3.8f), make_float2(2.740714286f, 3.814285714f), make_float2(2.751428571f, 3.828571429f),
    make_float2(2.762222222f, 3.842666667f), make_float2(2.773333333f, 3.856f), make_float2(2.784444444f, 3.869333333f),
    make_float2(2.795555556f, 3.882666667f), make_float2(2.806666667f, 3.896f), make_float2(2.817777778f, 3.909333333f),
    make_float2(2.828888889f, 3.922666667f), make_float2(2.84f, 3.936f), make_float2(2.851111111f, 3.949333333f),
    make_float2(2.862692308f, 3.960961538f), make_float2(2.876153846f, 3.965769231f), make_float2(2.889615385f, 3.970576923f),
    make_float2(2.903076923f, 3.975384615f), make_float2(2.916538462f, 3.980192308f), make_float2(2.93f, 3.985f),
    make_float2(2.943461538f, 3.989807692f), make_float2(2.956923077f, 3.994615385f), make_float2(2.970384615f, 3.999423077f),
    make_float2(2.983846154f, 4.004230769f), make_float2(2.997307692f, 4.009038462f), make_float2(3.012923077f, 4.01f),
    make_float2(3.029076923f, 4.01f), make_float2(3.045230769f, 4.01f), make_float2(3.061384615f, 4.01f), make_float2(3.077538462f, 4.01f),
    make_float2(3.093692308f, 4.01f), make_float2(3.109846154f, 4.01f), make_float2(3.126f, 4.01f), make_float2(3.142153846f, 4.01f),
    make_float2(3.158307692f, 4.01f), make_float2(3.174461538f, 4.01f), make_float2(3.190615385f, 4.01f), make_float2(3.206769231f, 4.01f),
    make_float2(3.214507042f, 4.007183099f), make_float2(3.220140845f, 4.003661972f)};

static constexpr std::array V{
    make_float2(2.582857143f, 3.365714286f), make_float2(2.643571429f, 3.387142857f), make_float2(2.709230769f, 3.407692308f),
    make_float2(2.782307692f, 3.426923077f), make_float2(2.855384615f, 3.446153846f), make_float2(2.92f, 3.4575f),
    make_float2(2.9825f, 3.466875f), make_float2(3.045f, 3.47625f), make_float2(3.115f, 3.481875f), make_float2(3.19f, 3.485f),
    make_float2(3.265f, 3.488125f), make_float2(3.333333333f, 3.49f), make_float2(3.391666667f, 3.49f), make_float2(3.45f, 3.49f),
    make_float2(3.508333333f, 3.49f), make_float2(3.534f, 3.484f), make_float2(3.5515f, 3.4765f), make_float2(3.569f, 3.469f),
    make_float2(3.5865f, 3.4615f), make_float2(3.634f, 3.444f), make_float2(3.689f, 3.424f), make_float2(3.744f, 3.404f),
    make_float2(3.799f, 3.384f), make_float2(3.8276f, 3.3608f), make_float2(3.8496f, 3.3368f), make_float2(3.8716f, 3.3128f),
    make_float2(3.8936f, 3.2888f), make_float2(3.9156f, 3.2648f), make_float2(3.9104f, 3.2472f), make_float2(3.8984f, 3.2312f),
    make_float2(3.8864f, 3.2152f), make_float2(3.8744f, 3.1992f), make_float2(3.8624f, 3.1832f), make_float2(3.88f, 3.16f),
    make_float2(3.905f, 3.135f), make_float2(3.93f, 3.11f), make_float2(3.955f, 3.085f), make_float2(3.98f, 3.06f),
    make_float2(3.994848485f, 3.038787879f), make_float2(3.969090909f, 3.032727273f), make_float2(3.943333333f, 3.026666667f),
    make_float2(3.917575758f, 3.020606061f), make_float2(3.891818182f, 3.014545455f), make_float2(3.866060606f, 3.008484848f),
    make_float2(3.84030303f, 3.002424242f), make_float2(3.805142857f, 3.001714286f), make_float2(3.763714286f, 3.004571429f),
    make_float2(3.722285714f, 3.007428571f), make_float2(3.680857143f, 3.010285714f), make_float2(3.639428571f, 3.013142857f),
    make_float2(3.598f, 3.016f), make_float2(3.556571429f, 3.018857143f), make_float2(3.519285714f, 3.025f),
    make_float2(3.484761905f, 3.033333333f), make_float2(3.450238095f, 3.041666667f), make_float2(3.415714286f, 3.05f),
    make_float2(3.381190476f, 3.058333333f), make_float2(3.346666667f, 3.066666667f), make_float2(3.312142857f, 3.075f),
    make_float2(3.277619048f, 3.083333333f), make_float2(3.248444444f, 3.091333333f), make_float2(3.240666667f, 3.098f),
    make_float2(3.232888889f, 3.104666667f), make_float2(3.225111111f, 3.111333333f), make_float2(3.217333333f, 3.118f),
    make_float2(3.209555556f, 3.124666667f), make_float2(3.201777778f, 3.131333333f), make_float2(3.194f, 3.138f),
    make_float2(3.186222222f, 3.144666667f), make_float2(3.180384615f, 3.150961538f), make_float2(3.182307692f, 3.155769231f),
    make_float2(3.184230769f, 3.160576923f), make_float2(3.186153846f, 3.165384615f), make_float2(3.188076923f, 3.170192308f),
    make_float2(3.19f, 3.175f), make_float2(3.191923077f, 3.179807692f), make_float2(3.193846154f, 3.184615385f),
    make_float2(3.195769231f, 3.189423077f), make_float2(3.197692308f, 3.194230769f), make_float2(3.199615385f, 3.199038462f),
    make_float2(3.197538462f, 3.203076923f), make_float2(3.194461538f, 3.206923077f), make_float2(3.191384615f, 3.210769231f),
    make_float2(3.188307692f, 3.214615385f), make_float2(3.185230769f, 3.218461538f), make_float2(3.182153846f, 3.222307692f),
    make_float2(3.179076923f, 3.226153846f), make_float2(3.176f, 3.23f), make_float2(3.172923077f, 3.233846154f),
    make_float2(3.169846154f, 3.237692308f), make_float2(3.166769231f, 3.241538462f), make_float2(3.163692308f, 3.245384615f),
    make_float2(3.160615385f, 3.249230769f), make_float2(3.157746479f, 3.255070423f), make_float2(3.154929577f, 3.261408451f)};

static constexpr std::array VN{
    make_float2(2.175093063f, 1.59177665f), make_float2(2.170862944f, 1.601928934f), make_float2(2.166632826f, 1.612081218f),
    make_float2(2.162402707f, 1.622233503f), make_float2(2.158172589f, 1.632385787f), make_float2(2.15394247f, 1.642538071f),
    make_float2(2.149712352f, 1.652690355f), make_float2(2.145482234f, 1.66284264f), make_float2(2.141252115f, 1.672994924f),
    make_float2(2.137021997f, 1.683147208f), make_float2(2.132791878f, 1.693299492f), make_float2(2.130823245f, 1.705762712f),
    make_float2(2.133244552f, 1.722711864f), make_float2(2.13566586f, 1.739661017f), make_float2(2.138087167f, 1.756610169f),
    make_float2(2.140508475f, 1.773559322f), make_float2(2.142929782f, 1.790508475f), make_float2(2.14535109f, 1.807457627f),
    make_float2(2.147772397f, 1.82440678f), make_float2(2.150193705f, 1.841355932f), make_float2(2.152615012f, 1.858305085f),
    make_float2(2.15503632f, 1.875254237f), make_float2(2.157457627f, 1.89220339f), make_float2(2.159878935f, 1.909152542f),
    make_float2(2.162300242f, 1.926101695f), make_float2(2.16472155f, 1.943050847f), make_float2(2.167142857f, 1.96f),
    make_float2(2.169564165f, 1.976949153f), make_float2(2.175951613f, 1.996201613f), make_float2(2.183209677f, 2.015959677f),
    make_float2(2.190467742f, 2.035717742f), make_float2(2.197725806f, 2.055475806f), make_float2(2.204983871f, 2.075233871f),
    make_float2(2.212241935f, 2.094991935f), make_float2(2.2195f, 2.11475f), make_float2(2.226758065f, 2.134508065f),
    make_float2(2.234016129f, 2.154266129f), make_float2(2.241274194f, 2.174024194f), make_float2(2.248532258f, 2.193782258f),
    make_float2(2.255790323f, 2.213540323f), make_float2(2.263048387f, 2.233298387f), make_float2(2.270306452f, 2.253056452f),
    make_float2(2.277564516f, 2.272814516f), make_float2(2.284822581f, 2.292572581f), make_float2(2.292080645f, 2.312330645f),
    make_float2(2.29933871f, 2.33208871f), make_float2(2.306596774f, 2.351846774f), make_float2(2.313854839f, 2.371604839f),
    make_float2(2.321112903f, 2.391362903f), make_float2(2.328370968f, 2.411120968f), make_float2(2.335629032f, 2.430879032f),
    make_float2(2.342887097f, 2.450637097f), make_float2(2.350183841f, 2.470217707f), make_float2(2.359375907f, 2.481103048f),
    make_float2(2.368567973f, 2.491988389f), make_float2(2.377760039f, 2.50287373f), make_float2(2.386952104f, 2.513759071f),
    make_float2(2.39614417f, 2.524644412f), make_float2(2.405336236f, 2.535529753f), make_float2(2.414528302f, 2.546415094f),
    make_float2(2.423720368f, 2.557300435f), make_float2(2.432912433f, 2.568185776f), make_float2(2.442104499f, 2.579071118f),
    make_float2(2.451296565f, 2.589956459f), make_float2(2.460488631f, 2.6008418f), make_float2(2.469680697f, 2.611727141f),
    make_float2(2.478872762f, 2.622612482f), make_float2(2.488064828f, 2.633497823f), make_float2(2.497256894f, 2.644383164f),
    make_float2(2.50644896f, 2.655268505f), make_float2(2.515641026f, 2.666153846f), make_float2(2.524833091f, 2.677039187f),
    make_float2(2.534025157f, 2.687924528f), make_float2(2.543217223f, 2.698809869f), make_float2(2.552409289f, 2.70969521f),
    make_float2(2.561601355f, 2.720580552f), make_float2(2.57079342f, 2.731465893f), make_float2(2.579985486f, 2.742351234f),
    make_float2(2.589177552f, 2.753236575f), make_float2(2.598369618f, 2.764121916f), make_float2(2.607561684f, 2.775007257f),
    make_float2(2.616753749f, 2.785892598f), make_float2(2.625945815f, 2.796777939f), make_float2(2.635137881f, 2.80766328f),
    make_float2(2.644329947f, 2.818548621f), make_float2(2.653522013f, 2.829433962f), make_float2(2.662714078f, 2.840319303f),
    make_float2(2.671906144f, 2.851204644f), make_float2(2.68109821f, 2.862089985f), make_float2(2.690290276f, 2.872975327f),
    make_float2(2.699482342f, 2.883860668f), make_float2(2.708674407f, 2.894746009f), make_float2(2.717866473f, 2.90563135f),
    make_float2(2.727058539f, 2.916516691f), make_float2(2.733784176f, 2.926087588f)};

static constexpr std::array Li{
    make_float2(0.3093694896f, 1.551618053f), make_float2(0.303494375f, 1.580608125f), make_float2(0.2954704779f, 1.607933493f),
    make_float2(0.2906108913f, 1.634534724f), make_float2(0.2839774394f, 1.659737734f), make_float2(0.277387047f, 1.683940839f),
    make_float2(0.2723517073f, 1.707465366f), make_float2(0.2681302362f, 1.730262677f), make_float2(0.2625129268f, 1.752152805f),
    make_float2(0.2558056098f, 1.783433293f), make_float2(0.2495064254f, 1.813613796f), make_float2(0.2433283452f, 1.839464736f),
    make_float2(0.237044465f, 1.857966876f), make_float2(0.2307925f, 1.892823191f), make_float2(0.2257989067f, 1.919315307f),
    make_float2(0.22209224f, 1.93620864f), make_float2(0.2168825225f, 1.96834955f), make_float2(0.2118802484f, 1.999772174f),
    make_float2(0.2071535404f, 2.030163478f), make_float2(0.201618503f, 2.059007365f), make_float2(0.1962138337f, 2.087298868f),
    make_float2(0.1923627945f, 2.114354296f), make_float2(0.1886596222f, 2.140744f), make_float2(0.1851540667f, 2.166244f),
    make_float2(0.1805051444f, 2.198260759f), make_float2(0.1751789412f, 2.234137765f), make_float2(0.1717437616f, 2.259857472f),
    make_float2(0.1689123124f, 2.282334347f), make_float2(0.1656800198f, 2.312320138f), make_float2(0.1623549209f, 2.344044447f),
    make_float2(0.1593099241f, 2.374155806f), make_float2(0.1563402846f, 2.403833226f), make_float2(0.1535770792f, 2.432204795f),
    make_float2(0.1509110191f, 2.459961847f), make_float2(0.148790401f, 2.486881665f), make_float2(0.1472559634f, 2.512901718f),
    make_float2(0.1458201169f, 2.540118815f), make_float2(0.1448101002f, 2.572506127f), make_float2(0.1438000835f, 2.604893439f),
    make_float2(0.1438082428f, 2.629614473f), make_float2(0.1440838019f, 2.652322141f), make_float2(0.1444524485f, 2.676618146f),
    make_float2(0.1450512281f, 2.704840877f), make_float2(0.1456500076f, 2.733063608f), make_float2(0.1460149235f, 2.764252629f),
    make_float2(0.1462625564f, 2.79692925f), make_float2(0.1465101894f, 2.82960587f), make_float2(0.1470538889f, 2.85453f),
    make_float2(0.1476233333f, 2.87878f), make_float2(0.1481807143f, 2.903183545f), make_float2(0.1484485714f, 2.931272169f),
    make_float2(0.1487164286f, 2.959360794f), make_float2(0.1489886038f, 2.98749756f), make_float2(0.1495263396f, 3.018595044f),
    make_float2(0.1500640755f, 3.049692528f), make_float2(0.1506018113f, 3.080790013f), make_float2(0.15112474f, 3.109828183f),
    make_float2(0.1516447639f, 3.138462373f), make_float2(0.1521647878f, 3.167096563f), make_float2(0.152889983f, 3.1946365f),
    make_float2(0.1538287238f, 3.221037521f), make_float2(0.1547674645f, 3.247438542f), make_float2(0.1557062053f, 3.273839563f),
    make_float2(0.1563975081f, 3.302050698f), make_float2(0.1570795704f, 3.330329431f), make_float2(0.1577616327f, 3.358608163f),
    make_float2(0.1584818293f, 3.387203049f), make_float2(0.1593227846f, 3.416799085f), make_float2(0.1601637398f, 3.446395122f),
    make_float2(0.1610046951f, 3.475991159f), make_float2(0.1617818234f, 3.505818676f), make_float2(0.1624128215f, 3.536176161f),
    make_float2(0.1630438196f, 3.566533647f), make_float2(0.1636748177f, 3.596891132f), make_float2(0.1643836923f, 3.626885538f),
    make_float2(0.1655873122f, 3.654573321f), make_float2(0.1667909321f, 3.682261104f), make_float2(0.167994552f, 3.709948887f),
    make_float2(0.1691981719f, 3.73763667f), make_float2(0.1700401022f, 3.765525434f), make_float2(0.170738569f, 3.793493918f),
    make_float2(0.1714370358f, 3.821462402f), make_float2(0.1721355026f, 3.849430886f), make_float2(0.17284576f, 3.87739856f),
    make_float2(0.17413376f, 3.90532656f), make_float2(0.17542176f, 3.93325456f), make_float2(0.17670976f, 3.96118256f),
    make_float2(0.17799776f, 3.98911056f), make_float2(0.1792768042f, 4.017031013f), make_float2(0.1801170143f, 4.04458165f),
    make_float2(0.1809572243f, 4.072132288f), make_float2(0.1817974344f, 4.099682926f), make_float2(0.1826376444f, 4.127233563f),
    make_float2(0.1834778545f, 4.154784201f), make_float2(0.1844895579f, 4.181916168f)};

}// namespace ior

using namespace luisa::compute;

class MetalSurface final : public Surface {

private:
    const Texture *_roughness;
    luisa::unique_ptr<Constant<float2>> _eta;
    bool _remap_roughness;

public:
    MetalSurface(Scene *scene, const SceneNodeDesc *desc) noexcept
        : Surface{scene, desc},
          _roughness{scene->load_texture(desc->property_node_or_default("roughness"))},
          _remap_roughness{desc->property_bool_or_default("remap_roughness", true)} {
        if (_roughness != nullptr && _roughness->category() != Texture::Category::GENERIC) [[unlikely]] {
            LUISA_ERROR(
                "Non-generic textures are not "
                "allowed in MetalSurface::roughness. [{}]",
                desc->source_location().string());
        }
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

public:
    MetalInstance(const Pipeline &pipeline, const Surface *surface, const Texture::Instance *roughness) noexcept
        : Surface::Instance{pipeline, surface}, _roughness{roughness} {}
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> closure(
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> MetalSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto roughness = pipeline.build_texture(command_buffer, _roughness);
    return luisa::make_unique<MetalInstance>(pipeline, this, roughness);
}

class MetalClosure final : public Surface::Closure {

private:
    FresnelConductor _fresnel;
    TrowbridgeReitzDistribution _distrib;
    MicrofacetReflection _lobe;

public:
    MetalClosure(
        const Surface::Instance *instance,
        const Interaction &it, const SampledWavelengths &swl, Expr<float> time,
        Expr<float4> n, Expr<float4> k, Expr<float2> alpha) noexcept
        : Surface::Closure{instance, it, swl, time},
          _fresnel{make_float4(1.f), n, k}, _distrib{alpha},
          _lobe{make_float4(1.f), &_distrib, &_fresnel} {}

private:
    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wi) const noexcept override {
        auto wo_local = _it.wo_local();
        auto wi_local = _it.shading().world_to_local(wi);
        auto f = _lobe.evaluate(wo_local, wi_local);
        auto pdf = _lobe.pdf(wo_local, wi_local);
        return {.swl = _swl,
                .f = f,
                .pdf = pdf,
                .alpha = _distrib.alpha(),
                .eta = make_float4(1.f)};
    }
    [[nodiscard]] Surface::Sample sample(Sampler::Instance &sampler) const noexcept override {
        auto wo_local = _it.wo_local();
        auto u = sampler.generate_2d();
        auto pdf = def(0.f);
        auto wi_local = def(make_float3(0.f, 0.f, 1.f));
        auto f = _lobe.sample(wo_local, &wi_local, u, &pdf);
        auto wi = _it.shading().local_to_world(wi_local);
        return {.wi = wi,
                .eval = {.swl = _swl,
                         .f = f,
                         .pdf = pdf,
                         .alpha = _distrib.alpha(),
                         .eta = make_float4(1.f)}};
    }
    void backward(Expr<float3> wi, Expr<float4> grad) const noexcept override {
    }
};

luisa::unique_ptr<Surface::Closure> MetalInstance::closure(const Interaction &it, const SampledWavelengths &swl, Expr<float> time) const noexcept {
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
    auto lambda = clamp(swl.lambda(), visible_wavelength_min, visible_wavelength_max);
    auto eta0 = sample_eta_k(lambda.x);
    auto eta1 = sample_eta_k(lambda.y);
    auto eta2 = sample_eta_k(lambda.z);
    auto eta3 = sample_eta_k(lambda.w);
    auto n = make_float4(eta0.x, eta1.x, eta2.x, eta3.x);
    auto k = make_float4(eta0.y, eta1.y, eta2.y, eta3.y);
    return luisa::make_unique<MetalClosure>(this, it, swl, time, n, k, alpha);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::MetalSurface)
