#include "stream_copy_mock.hpp"

QueueMock::QueueMock()
: m_basicQueueInfo {{0, 0}, INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE, TRAINING_QUEUE_COMPUTE_0, 0, nullptr},
  m_recordCounter(0),
  m_waitCounter(0),
  m_queryCounter(0),
  m_syncCounter(0),
  m_copyCounter(0),
  m_lastDirection(MEMCOPY_MAX_ENUM),
  m_lastIsUserRequest(true),
  m_pPreviousStream(nullptr)
{
}
