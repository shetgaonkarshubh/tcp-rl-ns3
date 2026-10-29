#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/opengym-module.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-socket-state.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <iomanip>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("RlTcpRandomDynamicSimulation");

// ============================================================================
// 1. OPEN GYM ENVIRONMENT DEFINITION
// ============================================================================
class TcpGymEnv : public OpenGymEnv
{
public:
  TcpGymEnv();
  virtual ~TcpGymEnv() override;
  static TypeId GetTypeId(void);

  virtual Ptr<OpenGymSpace> GetActionSpace(void) override;
  virtual Ptr<OpenGymSpace> GetObservationSpace(void) override;
  virtual Ptr<OpenGymDataContainer> GetObservation(void) override;
  virtual float GetReward(void) override;
  virtual bool GetGameOver(void) override;
  virtual bool ExecuteActions(Ptr<OpenGymDataContainer> action) override;
  virtual std::string GetExtraInfo(void) override;

  void StartStepTimer(Time stepTime);
  void UpdateSocketState(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, Time rtt);
  void RecordRtt(Time oldRtt, Time newRtt);
  void RecordDrop(void);

  void SetCurrentBottleneckMbps(double mbps) { m_currentBottleneckMbps = mbps; }
  double GetCurrentBottleneckMbps(void) const { return m_currentBottleneckMbps; }
  uint32_t GetTargetCwnd(void) const { return m_targetCwnd; }
  uint32_t GetTargetSsThresh(void) const { return m_targetSsThresh; }

private:
  void ScheduleNextStateRead(void);

  Time m_stepTime {Seconds(0.1)};
  uint32_t m_segmentSize {1448};
  uint32_t m_cWnd {14480};
  uint32_t m_ssThresh {52490};
  uint32_t m_targetCwnd {14480};
  uint32_t m_targetSsThresh {52490};

  uint32_t m_segmentsAcked {0};
  uint32_t m_bytesInFlight {0};
  Time m_rtt {MilliSeconds(42)};
  Time m_minRtt {MilliSeconds(42)};
  uint32_t m_congState {0};
  bool m_lossOccurred {false};
  bool m_isGameOver {false};
  double m_currentBottleneckMbps {10.0};

  Ptr<TcpSocketState> m_tcb {nullptr};
};

NS_OBJECT_ENSURE_REGISTERED(TcpGymEnv);

TypeId
TcpGymEnv::GetTypeId(void)
{
  static TypeId tid = TypeId("ns3::TcpGymEnv")
    .SetParent<OpenGymEnv>()
    .SetGroupName("OpenGym")
    .AddConstructor<TcpGymEnv>();
  return tid;
}

TcpGymEnv::TcpGymEnv() {}
TcpGymEnv::~TcpGymEnv() {}

void
TcpGymEnv::StartStepTimer(Time stepTime)
{
  m_stepTime = stepTime;
  Simulator::Schedule(Seconds(0.15), &TcpGymEnv::ScheduleNextStateRead, this);
}

void
TcpGymEnv::ScheduleNextStateRead(void)
{
  Notify();
  if (!Simulator::IsFinished())
    {
      Simulator::Schedule(m_stepTime, &TcpGymEnv::ScheduleNextStateRead, this);
    }
}

Ptr<OpenGymSpace>
TcpGymEnv::GetActionSpace(void)
{
  std::vector<uint32_t> shape = {2};
  return CreateObject<OpenGymBoxSpace>(0, 10000000, shape, TypeNameGet<uint32_t>());
}

Ptr<OpenGymSpace>
TcpGymEnv::GetObservationSpace(void)
{
  std::vector<uint32_t> shape = {15};
  return CreateObject<OpenGymBoxSpace>(0.0, 1e10, shape, TypeNameGet<double>());
}

Ptr<OpenGymDataContainer>
TcpGymEnv::GetObservation(void)
{
  std::vector<uint32_t> shape = {15};
  Ptr<OpenGymBoxContainer<double>> box = CreateObject<OpenGymBoxContainer<double>>(shape);

  uint32_t effectiveCongState = m_lossOccurred ? 3 : m_congState;

  box->AddValue(0.0);                                            // [0] socketUuid
  box->AddValue(1.0);                                            // [1] envType
  box->AddValue(Simulator::Now().GetMicroSeconds());             // [2] simTime_us
  box->AddValue(0.0);                                            // [3] nodeId
  box->AddValue(static_cast<double>(m_ssThresh));                // [4] ssThresh (bytes)
  box->AddValue(static_cast<double>(m_cWnd));                    // [5] cWnd (bytes)
  box->AddValue(static_cast<double>(m_segmentSize));             // [6] segmentSize (1448 B)
  box->AddValue(static_cast<double>(m_segmentsAcked));           // [7] segmentsAcked
  box->AddValue(static_cast<double>(m_bytesInFlight));           // [8] bytesInFlight
  box->AddValue(static_cast<double>(m_rtt.GetMicroSeconds()));   // [9] lastRtt_us
  box->AddValue(static_cast<double>(m_minRtt.GetMicroSeconds()));// [10] minRtt_us
  box->AddValue(0.0);                                            // [11] calledFunc
  box->AddValue(static_cast<double>(effectiveCongState));        // [12] congState
  box->AddValue(m_currentBottleneckMbps);                        // [13] active bottleneck Mbps
  box->AddValue(0.0);                                            // [14] ecnState

  m_segmentsAcked = 0;
  m_lossOccurred = false;

  return box;
}

float
TcpGymEnv::GetReward(void)
{
  return 0.0f;
}

bool
TcpGymEnv::GetGameOver(void)
{
  return m_isGameOver;
}

std::string
TcpGymEnv::GetExtraInfo(void)
{
  return std::to_string(m_currentBottleneckMbps);
}

bool
TcpGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
  Ptr<OpenGymBoxContainer<uint32_t>> box = DynamicCast<OpenGymBoxContainer<uint32_t>>(action);
  if (box && box->GetData().size() >= 2)
    {
      uint32_t newSsThresh = box->GetValue(0);
      uint32_t newCwnd = box->GetValue(1);

      uint32_t lowerBound = 2 * m_segmentSize;
      uint32_t upperBound = 200 * m_segmentSize;

      m_targetCwnd = std::max(lowerBound, std::min(newCwnd, upperBound));
      m_targetSsThresh = std::max(lowerBound, std::min(newSsThresh, upperBound));
      m_cWnd = m_targetCwnd;
      m_ssThresh = m_targetSsThresh;

      if (m_tcb)
        {
          m_tcb->m_cWnd = m_targetCwnd;
          m_tcb->m_ssThresh = m_targetSsThresh;
        }
    }
  return true;
}

void
TcpGymEnv::UpdateSocketState(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, Time rtt)
{
  m_tcb = tcb;
  m_segmentSize = tcb->m_segmentSize;
  m_segmentsAcked += segmentsAcked;
  m_bytesInFlight = tcb->m_bytesInFlight.Get();
  m_congState = static_cast<uint32_t>(tcb->m_congState.Get());

  if (rtt > Seconds(0.0))
    {
      RecordRtt(m_rtt, rtt);
    }

  tcb->m_cWnd = m_targetCwnd;
  tcb->m_ssThresh = m_targetSsThresh;
  m_cWnd = m_targetCwnd;
  m_ssThresh = m_targetSsThresh;
}

void
TcpGymEnv::RecordRtt(Time oldRtt, Time newRtt)
{
  if (newRtt > Seconds(0.0))
    {
      m_rtt = newRtt;
      if (newRtt < m_minRtt)
        {
          m_minRtt = newRtt;
        }
    }
}

void
TcpGymEnv::RecordDrop(void)
{
  m_lossOccurred = true;
}

// ============================================================================
// 2. CUSTOM RL CONGESTION OPS
// ============================================================================
static Ptr<TcpGymEnv> g_gymEnv = nullptr;
static std::mt19937 g_rng;

class TcpRlDynamicCongestion : public TcpNewReno
{
public:
  static TypeId GetTypeId(void)
  {
    static TypeId tid = TypeId("ns3::TcpRlDynamicCongestion")
      .SetParent<TcpNewReno>()
      .SetGroupName("Internet")
      .AddConstructor<TcpRlDynamicCongestion>();
    return tid;
  }

  TcpRlDynamicCongestion() : TcpNewReno() {}
  TcpRlDynamicCongestion(const TcpRlDynamicCongestion& sock) : TcpNewReno(sock) {}
  virtual ~TcpRlDynamicCongestion() override {}

  virtual std::string GetName() const override { return "TcpRlDynamicCongestion"; }

  virtual void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked,
                         const Time& rtt) override
  {
    if (g_gymEnv)
      {
        g_gymEnv->UpdateSocketState(tcb, segmentsAcked, rtt);
      }
  }

  virtual void IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override
  {
    if (g_gymEnv)
      {
        tcb->m_cWnd = g_gymEnv->GetTargetCwnd();
        tcb->m_ssThresh = g_gymEnv->GetTargetSsThresh();
      }
  }

  virtual uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) override
  {
    if (g_gymEnv)
      {
        return g_gymEnv->GetTargetSsThresh();
      }
    return TcpNewReno::GetSsThresh(tcb, bytesInFlight);
  }

  virtual Ptr<TcpCongestionOps> Fork() override
  {
    return CopyObject<TcpRlDynamicCongestion>(this);
  }
};

NS_OBJECT_ENSURE_REGISTERED(TcpRlDynamicCongestion);

// ============================================================================
// 3. STOCHASTIC BOTTLENECK SCHEDULER (Random Rate + Random Hold Duration)
// ============================================================================
static void
DropTrace(Ptr<TcpGymEnv> env, Ptr<const Packet> p)
{
  env->RecordDrop();
}

static void
RandomizeBottleneckBandwidth(NetDeviceContainer bottleneckDevs, Ptr<TcpGymEnv> env)
{
  // 1. Sample a random bandwidth in [3.5, 11.5] Mbps (BDP in [12.7, 41.7] MSS)
  // Enforce a minimum jump of >= 2.0 Mbps from the current rate so every transition
  // forces the RL agent to actively adapt.
  std::uniform_real_distribution<double> rateDist(3.5, 11.5);
  std::uniform_real_distribution<double> holdTimeDist(3.5, 9.0); // 35 to 90 steps

  double prevMbps = env ? env->GetCurrentBottleneckMbps() : 10.0;
  double newMbps = rateDist(g_rng);
  for (int tries = 0; tries < 10 && std::abs(newMbps - prevMbps) < 2.0; ++tries)
    {
      newMbps = rateDist(g_rng);
    }
  // Round to 1 decimal place (e.g., "7.4Mbps")
  newMbps = std::round(newMbps * 10.0) / 10.0;

  std::ostringstream rateStream;
  rateStream << std::fixed << std::setprecision(1) << newMbps << "Mbps";
  DataRate newRate(rateStream.str());

  Ptr<PointToPointNetDevice> dev0 = DynamicCast<PointToPointNetDevice>(bottleneckDevs.Get(0));
  Ptr<PointToPointNetDevice> dev1 = DynamicCast<PointToPointNetDevice>(bottleneckDevs.Get(1));
  if (dev0 && dev1)
    {
      dev0->SetDataRate(newRate);
      dev1->SetDataRate(newRate);
    }

  if (env)
    {
      env->SetCurrentBottleneckMbps(newMbps);
    }

  // 2. Schedule the NEXT change after a completely random duration in [3.5s, 9.0s]
  double nextHoldSeconds = holdTimeDist(g_rng);
  if (!Simulator::IsFinished())
    {
      Simulator::Schedule(Seconds(nextHoldSeconds),
                          &RandomizeBottleneckBandwidth,
                          bottleneckDevs,
                          env);
    }
}

// ============================================================================
// 4. MAIN FUNCTION
// ============================================================================
int main(int argc, char *argv[])
{
  uint32_t openGymPort = 7144;
  double simTime = 15000.0;
  double envStepTime = 0.1;
  uint32_t randomSeed = 0; // 0 = use hardware random_device

  CommandLine cmd;
  cmd.AddValue("openGymPort", "Port for OpenGym IPC", openGymPort);
  cmd.AddValue("simTime", "Simulation time in seconds", simTime);
  cmd.AddValue("duration", "Simulation time in seconds (alias)", simTime);
  cmd.AddValue("seed", "RNG seed (0 for true hardware randomness)", randomSeed);
  cmd.Parse(argc, argv);

  if (randomSeed == 0)
    {
      std::random_device rd;
      g_rng.seed(rd());
    }
  else
    {
      g_rng.seed(randomSeed);
    }

  Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1448));
  Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
  Config::SetDefault("ns3::TcpL4Protocol::SocketType",
                     TypeIdValue(TcpRlDynamicCongestion::GetTypeId()));

  NodeContainer senders, routers, receivers;
  senders.Create(1);
  routers.Create(2);
  receivers.Create(1);

  // 100 Mbps, 0.5ms access links -> 42.0ms total unloaded RTT
  PointToPointHelper accessLink;
  accessLink.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
  accessLink.SetChannelAttribute("Delay", StringValue("0.5ms"));

  // Stochastic Bottleneck Link (20ms one-way delay, 60-packet DropTail queue)
  PointToPointHelper bottleneckLink;
  bottleneckLink.SetDeviceAttribute("DataRate", StringValue("10Mbps"));
  bottleneckLink.SetChannelAttribute("Delay", StringValue("20ms"));
  bottleneckLink.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", StringValue("60p"));

  NetDeviceContainer devSenderRouter = accessLink.Install(senders.Get(0), routers.Get(0));
  NetDeviceContainer devBottleneck = bottleneckLink.Install(routers.Get(0), routers.Get(1));
  NetDeviceContainer devRouterReceiver = accessLink.Install(routers.Get(1), receivers.Get(0));

  InternetStackHelper stack;
  stack.Install(senders);
  stack.Install(routers);
  stack.Install(receivers);

  Ipv4AddressHelper address;
  address.SetBase("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer ifSenderRouter = address.Assign(devSenderRouter);

  address.SetBase("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer ifBottleneck = address.Assign(devBottleneck);

  address.SetBase("10.1.3.0", "255.255.255.0");
  Ipv4InterfaceContainer ifRouterReceiver = address.Assign(devRouterReceiver);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables();

  Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface>(openGymPort);
  g_gymEnv = CreateObject<TcpGymEnv>();
  g_gymEnv->SetOpenGymInterface(openGymInterface);
  g_gymEnv->StartStepTimer(Seconds(envStepTime));

  Ptr<PointToPointNetDevice> bnDev = DynamicCast<PointToPointNetDevice>(devBottleneck.Get(0));
  if (bnDev && bnDev->GetQueue())
    {
      bnDev->GetQueue()->TraceConnectWithoutContext(
        "Drop", MakeBoundCallback(&DropTrace, g_gymEnv));
    }

  // Start the random bottleneck scheduler
  Simulator::Schedule(Seconds(0.0), &RandomizeBottleneckBandwidth, devBottleneck, g_gymEnv);

  uint16_t sinkPort = 8080;
  Address sinkAddress(InetSocketAddress(ifRouterReceiver.GetAddress(1), sinkPort));
  PacketSinkHelper packetSinkHelper("ns3::TcpSocketFactory",
                                    InetSocketAddress(Ipv4Address::GetAny(), sinkPort));
  ApplicationContainer sinkApps = packetSinkHelper.Install(receivers.Get(0));
  sinkApps.Start(Seconds(0.0));
  sinkApps.Stop(Seconds(simTime));

  BulkSendHelper sourceHelper("ns3::TcpSocketFactory", sinkAddress);
  sourceHelper.SetAttribute("MaxBytes", UintegerValue(0));
  ApplicationContainer sourceApps = sourceHelper.Install(senders.Get(0));
  sourceApps.Start(Seconds(0.1));
  sourceApps.Stop(Seconds(simTime));

  Simulator::Stop(Seconds(simTime));
  Simulator::Run();
  openGymInterface->NotifySimulationEnd();
  Simulator::Destroy();
  return 0;
}