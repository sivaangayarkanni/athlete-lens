# Athlete Lens — product note

## User

A college / SDAT / school coach with 12–40 athletes, a phone, and a register.

## Jobs to be done

1. “Is this athlete safe to do speed work tomorrow?”
2. “Who on my squad is quietly overreaching?”
3. “What do I change this week without a lab?”

## Inputs that actually exist in India

- 30m / 100m from a phone stopwatch
- standing long jump or vertical jump against a wall
- session minutes
- RPE (taught in 30 seconds)
- sleep hours
- rest days

Heart-rate is optional. The model still runs without it.

## Decision outputs

- Green / amber / red readiness
- Injury-risk band + probability
- Four concrete recommendations with timing (`now` / `this_week` / `monitor`)

## Next production steps

- Auth (coach vs athlete)
- Postgres + multi-academy tenancy
- WhatsApp digest every Sunday night
- Retrain hook when a club crosses 200 labelled sessions
